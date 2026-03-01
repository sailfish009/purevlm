"""
Pure VLM (NumPy): BAE × BAE + Tensor Product Attention
========================================================

Two innovations:
  1. BAE × BAE visual tokenizer (from previous)
  2. Tensor Product Attention (NEW)

Standard attention:
  score(q, k) = q^T k / √d                    — degree-2 kernel

Tensor Product Attention:
  score(q, k) = (q₁^T k₁)(q₂^T k₂) / d      — degree-4 kernel
  
  Each head's q,k are split in half:
    q = [q₁ | q₂],  k = [k₁ | k₂]
    score = (q₁·k₁) × (q₂·k₂)
  
  This captures "feature-group-A matches AND feature-group-B matches"
  rather than just "overall similarity". Critical for distinguishing
  digits like 8 vs 0 where top-half is similar but bottom-half differs.
  
  No extra parameters — just changes how scores are computed.

  Backward: d/dq₁ = (q₂·k₂) × k₁,  d/dk₁ = (q₂·k₂) × q₁  (and symmetric)
"""

import os
import struct
import gzip
import time
import numpy as np

np.random.seed(42)

# =============================================================================
# MNIST Loader
# =============================================================================
def load_mnist(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(n), dtype=np.uint8)
    with gzip.open(images_path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, rows * cols) / 255.0
    return images, labels

def download_mnist():
    import urllib.request
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    os.makedirs('mnist', exist_ok=True)
    for fname in files:
        path = os.path.join('mnist', fname)
        if not os.path.exists(path):
            print(f"downloading {fname}...")
            urllib.request.urlretrieve(base + fname, path)

download_mnist()
train_images, train_labels = load_mnist(
    'mnist/train-images-idx3-ubyte.gz', 
    'mnist/train-labels-idx1-ubyte.gz'
)
print(f"loaded {len(train_images)} training images, 28x28={train_images.shape[1]}px")

# =============================================================================
# Tokenizer
# =============================================================================
digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
text_chars = sorted(set(''.join(digit_names)))
BOS = len(text_chars)
EOS = len(text_chars) + 1
text_vocab_size = len(text_chars) + 2
char_to_id = {ch: i for i, ch in enumerate(text_chars)}
print(f"text vocab: {text_chars} + BOS + EOS = {text_vocab_size} tokens")

# =============================================================================
# Downscale
# =============================================================================
IMG_SIZE = 14
N_PIXELS = IMG_SIZE * IMG_SIZE

def downscale_all(images, orig=28, target=IMG_SIZE):
    n = len(images)
    imgs = images.reshape(n, orig, orig)
    out = np.zeros((n, target, target))
    for r in range(target):
        for c in range(target):
            r0, r1 = int(r * orig / target), int((r+1) * orig / target)
            c0, c1 = int(c * orig / target), int((c+1) * orig / target)
            out[:, r, c] = imgs[:, r0:r1, c0:c1].mean(axis=(1, 2))
    return out.reshape(n, target * target)

print(f"downscaling images to {IMG_SIZE}x{IMG_SIZE}...")
train_images_small = downscale_all(train_images)

# =============================================================================
# Hyperparameters
# =============================================================================
n_layer = 1
n_embd = 32
n_head = 4
head_dim = n_embd // n_head

# --- BAE × BAE Configuration ---
# Two base BAEs (degree-2 each)
bae_base_dim = 10       # Each base BAE produces 10 features

# Tensor product tokens: low-rank samples from bae_base_dim² space
n_tensor_tokens = 12    # degree-4 tokens from tensor product

# Also keep direct degree-2 and degree-3 features (skip connection)
bae_degrees = [2, 3]
tokens_per_degree = [8, 8]  # 8 direct tokens per degree
n_direct_tokens = sum(tokens_per_degree)

# Total visual tokens
n_vis_tokens = n_tensor_tokens + n_direct_tokens  # 12 + 16 = 28
max_text_len = 8
block_size = n_vis_tokens + max_text_len

print(f"\n=== BAE × BAE Configuration ===")
print(f"BAE₁: {bae_base_dim} features (degree-2)")
print(f"BAE₂: {bae_base_dim} features (degree-2)")
print(f"Tensor product: {bae_base_dim}²={bae_base_dim**2} implicit dims → {n_tensor_tokens} tokens (degree-4)")
print(f"Direct features: {tokens_per_degree} (degree {bae_degrees})")
print(f"Total vision tokens: {n_vis_tokens}")

# =============================================================================
# Parameters & Gradients
# =============================================================================
def param(shape, std=0.08):
    return np.random.randn(*shape).astype(np.float64) * std

P = {}
G = {}

# --- BAE × BAE: two base BAEs ---
P['bae1_l'] = param((bae_base_dim, N_PIXELS), 0.05)
P['bae1_r'] = param((bae_base_dim, N_PIXELS), 0.05)
P['bae2_l'] = param((bae_base_dim, N_PIXELS), 0.05)
P['bae2_r'] = param((bae_base_dim, N_PIXELS), 0.05)

# --- Tensor product projection: sample rank-k from m² space ---
# token[j] = (P_a[j] · f₁) × (P_b[j] · f₂)
P['tp_proj_a'] = param((n_tensor_tokens, bae_base_dim), 0.08)
P['tp_proj_b'] = param((n_tensor_tokens, bae_base_dim), 0.08)

# --- Direct multi-degree BAE (skip connection) ---
for deg_idx, degree in enumerate(bae_degrees):
    n_tok = tokens_per_degree[deg_idx]
    for d in range(degree):
        P[f'bae_deg{degree}_{d}'] = param((n_tok, N_PIXELS), 0.05)

# --- Vision projection (bilinear, from your working code) ---
P['vis_proj_l'] = param((n_embd, n_vis_tokens), 0.08)
P['vis_proj_r'] = param((n_embd, n_vis_tokens), 0.08)

# --- Embeddings ---
P['wte'] = param((text_vocab_size, n_embd))
P['wpe'] = param((block_size, n_embd))

# --- Transformer ---
for i in range(n_layer):
    P[f'L{i}.wq'] = param((n_embd, n_embd))
    P[f'L{i}.wk'] = param((n_embd, n_embd))
    P[f'L{i}.wv'] = param((n_embd, n_embd))
    P[f'L{i}.wo'] = param((n_embd, n_embd))
    P[f'L{i}.mlp_fc1'] = param((4 * n_embd, n_embd))
    P[f'L{i}.mlp_fc2'] = param((n_embd, 4 * n_embd))

P['lm_head'] = param((text_vocab_size, n_embd))

for k in P:
    G[k] = np.zeros_like(P[k])

n_params = sum(p.size for p in P.values())
print(f"\nnum params: {n_params:,}")
print(f"architecture: BAE×BAE + Tensor Product Attention + GPT")
print(f"attention: score = (q₁·k₁)(q₂·k₂)/d  [degree-4 kernel, no extra params]")

# =============================================================================
# Primitives
# =============================================================================
def rmsnorm(x, eps=1e-5):
    ms = np.mean(x ** 2) + eps
    return x / np.sqrt(ms), np.sqrt(ms)

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

# =============================================================================
# BAE × BAE Encode
# =============================================================================
def bae_encode(x_norm, cache):
    """
    BAE×BAE tensor product + multi-degree skip features.
    
    1. Two base BAEs → f₁, f₂ (each degree-2, independently normalized)
    2. Tensor product tokens: token[j] = (P_a[j]·f₁) × (P_b[j]·f₂)
    3. Direct multi-degree features (degree-2, degree-3 skip)
    4. Concatenate all → normalize
    """
    
    # --- Base BAE₁ and BAE₂ ---
    l1x = P['bae1_l'] @ x_norm
    r1x = P['bae1_r'] @ x_norm
    f1_raw = l1x * r1x
    f1_norm = np.linalg.norm(f1_raw) + 1e-8
    f1 = f1_raw / f1_norm
    
    l2x = P['bae2_l'] @ x_norm
    r2x = P['bae2_r'] @ x_norm
    f2_raw = l2x * r2x
    f2_norm = np.linalg.norm(f2_raw) + 1e-8
    f2 = f2_raw / f2_norm
    
    cache['l1x'], cache['r1x'] = l1x, r1x
    cache['l2x'], cache['r2x'] = l2x, r2x
    cache['f1_raw'], cache['f1_norm'], cache['f1'] = f1_raw, f1_norm, f1
    cache['f2_raw'], cache['f2_norm'], cache['f2'] = f2_raw, f2_norm, f2
    
    # --- Tensor product tokens (degree-4 via kernel trick) ---
    # a[j] = P_a[j] · f₁,  b[j] = P_b[j] · f₂
    # tensor_tok[j] = a[j] × b[j]
    a = P['tp_proj_a'] @ f1    # (n_tensor_tokens,)
    b = P['tp_proj_b'] @ f2    # (n_tensor_tokens,)
    tensor_tokens = a * b       # (n_tensor_tokens,) — degree-4!
    
    cache['tp_a'], cache['tp_b'] = a, b
    cache['tensor_tokens'] = tensor_tokens
    
    # --- Direct multi-degree skip features ---
    f_direct_parts = []
    for deg_idx, degree in enumerate(bae_degrees):
        n_tok = tokens_per_degree[deg_idx]
        projections = []
        for d in range(degree):
            proj = P[f'bae_deg{degree}_{d}'] @ x_norm
            projections.append(proj)
            cache[f'proj_deg{degree}_{d}'] = proj
        
        f_degree = np.ones(n_tok)
        for proj in projections:
            f_degree = f_degree * proj
        
        f_direct_parts.append(f_degree)
        cache[f'f_deg{degree}'] = f_degree
    
    f_direct = np.concatenate(f_direct_parts)
    
    # --- Concatenate: tensor product + direct ---
    f_all = np.concatenate([tensor_tokens, f_direct])
    
    # --- Normalize ---
    f_all_norm = np.linalg.norm(f_all) + 1e-8
    f = f_all / f_all_norm
    
    cache['f_direct'] = f_direct
    cache['f_all'] = f_all
    cache['f_all_norm'] = f_all_norm
    cache['f'] = f
    
    return f

# =============================================================================
# Reconstruction Loss (degree-2 only, stable)
# =============================================================================
def compute_recon_loss(cache, x_norm):
    f_deg2 = cache.get('f_deg2', None)
    if f_deg2 is None:
        return 0.0
    
    f_n = f_deg2 / (np.linalg.norm(f_deg2) + 1e-8)
    L = P['bae_deg2_0']
    R = P['bae_deg2_1']
    L_n = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    R_n = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    BB = (L_n @ L_n.T) * (R_n @ R_n.T)
    recon = f_n @ BB @ f_n - 2 * (f_n @ f_n) + (x_norm @ x_norm)
    return np.clip(recon, 0, 10.0)

# =============================================================================
# Forward + Backward
# =============================================================================
def forward_backward(idx, text_tokens, alpha_recon=0.01):
    for k in G:
        G[k][:] = 0.0
    
    cache = {}
    seq_len = n_vis_tokens + len(text_tokens)
    
    x_raw = train_images_small[idx].copy()
    x_norm = x_raw / (np.sqrt(np.sum(x_raw ** 2)) + 1e-8)
    cache['x_norm'] = x_norm
    
    # =========================================================================
    # FORWARD
    # =========================================================================
    f = bae_encode(x_norm, cache)
    recon_loss = compute_recon_loss(cache, x_norm)
    
    # Vision projection (bilinear)
    vl = P['vis_proj_l'] @ f
    vr = P['vis_proj_r'] @ f
    vis_emb_base = vl * vr
    cache['vis_vl'], cache['vis_vr'] = vl, vr
    
    # Build embeddings
    embeddings = np.zeros((seq_len, n_embd))
    for pos in range(n_vis_tokens):
        embeddings[pos] = vis_emb_base + P['wpe'][pos]
    for t, tok in enumerate(text_tokens):
        pos = n_vis_tokens + t
        embeddings[pos] = P['wte'][tok] + P['wpe'][pos]
    
    # Initial RMSNorm
    normed_embs = np.zeros_like(embeddings)
    rms_scales_init = np.zeros(seq_len)
    for i in range(seq_len):
        normed_embs[i], rms_scales_init[i] = rmsnorm(embeddings[i])
    cache['rms_scales_init'] = rms_scales_init
    
    # Transformer
    for li in range(n_layer):
        x_seq = normed_embs.copy()
        
        x_ln = np.zeros_like(x_seq)
        rms_scales_attn = np.zeros(seq_len)
        for i in range(seq_len):
            x_ln[i], rms_scales_attn[i] = rmsnorm(x_seq[i])
        cache[f'L{li}_x_ln'] = x_ln
        cache[f'L{li}_rms_attn'] = rms_scales_attn
        
        Q = x_ln @ P[f'L{li}.wq'].T
        K = x_ln @ P[f'L{li}.wk'].T
        V = x_ln @ P[f'L{li}.wv'].T
        cache[f'L{li}_Q'], cache[f'L{li}_K'], cache[f'L{li}_V'] = Q, K, V
        
        attn_out = np.zeros((seq_len, n_embd))
        attn_weights_cache = []
        
        half = head_dim // 2  # Split each head into two halves
        
        for i in range(seq_len):
            head_outputs = []
            head_weights = []
            for h in range(n_head):
                hs, he = h * head_dim, (h + 1) * head_dim
                hm = hs + half  # midpoint
                
                # Split q into [q1 | q2], k into [k1 | k2]
                q1 = Q[i, hs:hm]
                q2 = Q[i, hm:he]
                k1 = K[:i+1, hs:hm]
                k2 = K[:i+1, hm:he]
                v_h = V[:i+1, hs:he]
                
                # Tensor product score: (q1·k1) × (q2·k2) — degree-4
                scores1 = k1 @ q1      # (i+1,)
                scores2 = k2 @ q2      # (i+1,)
                scores = (scores1 * scores2) / head_dim
                
                w = softmax(scores)
                head_weights.append((w, scores1, scores2))  # Cache both for backward
                head_outputs.append(w @ v_h)
            attn_out[i] = np.concatenate(head_outputs)
            attn_weights_cache.append(head_weights)
        
        cache[f'L{li}_attn_weights'] = attn_weights_cache
        cache[f'L{li}_attn_out'] = attn_out
        
        attn_proj = attn_out @ P[f'L{li}.wo'].T
        x_after_attn = x_seq + attn_proj
        cache[f'L{li}_x_after_attn'] = x_after_attn
        
        x_mlp_ln = np.zeros_like(x_after_attn)
        rms_scales_mlp = np.zeros(seq_len)
        for i in range(seq_len):
            x_mlp_ln[i], rms_scales_mlp[i] = rmsnorm(x_after_attn[i])
        cache[f'L{li}_x_mlp_ln'] = x_mlp_ln
        cache[f'L{li}_rms_mlp'] = rms_scales_mlp
        
        mlp_hidden = x_mlp_ln @ P[f'L{li}.mlp_fc1'].T
        mlp_hidden_relu = np.maximum(0, mlp_hidden)
        mlp_out = mlp_hidden_relu @ P[f'L{li}.mlp_fc2'].T
        
        normed_embs = x_after_attn + mlp_out
        cache[f'L{li}_mlp_hidden'] = mlp_hidden
        cache[f'L{li}_mlp_hidden_relu'] = mlp_hidden_relu
    
    # LM Head
    text_losses = []
    all_probs = []
    n_text = len(text_tokens) - 1
    
    for t in range(n_text):
        pos = n_vis_tokens + t
        logits = P['lm_head'] @ normed_embs[pos]
        probs = softmax(logits)
        target = text_tokens[t + 1]
        text_losses.append(-np.log(probs[target] + 1e-10))
        all_probs.append((probs.copy(), target, pos))
    
    text_loss = np.mean(text_losses)
    total_loss = text_loss + alpha_recon * recon_loss
    
    cache['normed_embs_final'] = normed_embs
    cache['all_probs'] = all_probs
    cache['n_text'] = n_text
    cache['seq_len'] = seq_len
    
    # =========================================================================
    # BACKWARD
    # =========================================================================
    
    # LM Head
    d_final = np.zeros_like(normed_embs)
    for probs, target, pos in all_probs:
        d_logits = probs.copy()
        d_logits[target] -= 1.0
        d_logits /= n_text
        G['lm_head'] += np.outer(d_logits, normed_embs[pos])
        d_final[pos] += P['lm_head'].T @ d_logits
    
    # Transformer backward
    d_x = d_final.copy()
    
    for li in range(n_layer - 1, -1, -1):
        d_mlp_out = d_x.copy()
        G[f'L{li}.mlp_fc2'] += d_mlp_out.T @ cache[f'L{li}_mlp_hidden_relu']
        d_mlp_hidden_relu = d_mlp_out @ P[f'L{li}.mlp_fc2']
        d_mlp_hidden = d_mlp_hidden_relu * (cache[f'L{li}_mlp_hidden'] > 0)
        
        x_mlp_ln = cache[f'L{li}_x_mlp_ln']
        G[f'L{li}.mlp_fc1'] += d_mlp_hidden.T @ x_mlp_ln
        d_x_mlp_ln = d_mlp_hidden @ P[f'L{li}.mlp_fc1']
        
        rms_mlp = cache[f'L{li}_rms_mlp']
        d_x_after_attn = d_x.copy()
        for i in range(seq_len):
            d_x_after_attn[i] += d_x_mlp_ln[i] * rms_mlp[i]
        
        attn_out_cache = cache[f'L{li}_attn_out']
        G[f'L{li}.wo'] += d_x_after_attn.T @ attn_out_cache
        d_attn_out = d_x_after_attn @ P[f'L{li}.wo']
        
        d_x_seq = d_x_after_attn.copy()
        Q, K, V = cache[f'L{li}_Q'], cache[f'L{li}_K'], cache[f'L{li}_V']
        all_weights = cache[f'L{li}_attn_weights']
        d_Q, d_K, d_V = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
        
        half = head_dim // 2
        
        for i in range(seq_len):
            for h in range(n_head):
                hs, he = h * head_dim, (h + 1) * head_dim
                hm = hs + half
                
                w, scores1, scores2 = all_weights[i][h]
                d_head_out = d_attn_out[i, hs:he]
                v_h = V[:i+1, hs:he]
                
                # d_w, d_V (same as standard)
                d_w = v_h @ d_head_out
                for t in range(i + 1):
                    d_V[t, hs:he] += w[t] * d_head_out
                
                # Through softmax
                d_scores = w * (d_w - np.dot(w, d_w))
                
                # Through tensor product: score = (s1 * s2) / head_dim
                # d_s1 = d_scores * s2 / head_dim
                # d_s2 = d_scores * s1 / head_dim
                d_s1 = d_scores * scores2 / head_dim
                d_s2 = d_scores * scores1 / head_dim
                
                # s1[t] = k1[t] · q1,  s2[t] = k2[t] · q2
                q1 = Q[i, hs:hm]
                q2 = Q[i, hm:he]
                
                for t in range(i + 1):
                    k1_t = K[t, hs:hm]
                    k2_t = K[t, hm:he]
                    
                    d_Q[i, hs:hm] += d_s1[t] * k1_t
                    d_Q[i, hm:he] += d_s2[t] * k2_t
                    d_K[t, hs:hm] += d_s1[t] * q1
                    d_K[t, hm:he] += d_s2[t] * q2
        
        x_ln = cache[f'L{li}_x_ln']
        G[f'L{li}.wq'] += d_Q.T @ x_ln
        G[f'L{li}.wk'] += d_K.T @ x_ln
        G[f'L{li}.wv'] += d_V.T @ x_ln
        d_x_ln = d_Q @ P[f'L{li}.wq'] + d_K @ P[f'L{li}.wk'] + d_V @ P[f'L{li}.wv']
        
        rms_attn = cache[f'L{li}_rms_attn']
        for i in range(seq_len):
            d_x_seq[i] += d_x_ln[i] * rms_attn[i]
        d_x = d_x_seq
    
    # Initial RMSNorm backward
    rms_init = cache['rms_scales_init']
    d_emb = np.zeros((seq_len, n_embd))
    for i in range(seq_len):
        d_emb[i] = d_x[i] * rms_init[i]
    
    # Embedding backward
    d_vis_emb_base = np.zeros(n_embd)
    for pos in range(n_vis_tokens):
        d_vis_emb_base += d_emb[pos]
        G['wpe'][pos] += d_emb[pos]
    for t, tok in enumerate(text_tokens):
        pos = n_vis_tokens + t
        G['wte'][tok] += d_emb[pos]
        G['wpe'][pos] += d_emb[pos]
    
    # Vision projection backward (bilinear)
    f = cache['f']
    vl, vr = cache['vis_vl'], cache['vis_vr']
    G['vis_proj_l'] += np.outer(d_vis_emb_base * vr, f)
    G['vis_proj_r'] += np.outer(d_vis_emb_base * vl, f)
    d_f = P['vis_proj_l'].T @ (d_vis_emb_base * vr) + \
          P['vis_proj_r'].T @ (d_vis_emb_base * vl)
    
    # =========================================================================
    # BAE × BAE Backward
    # =========================================================================
    
    # Through global normalization
    f_all = cache['f_all']
    f_all_norm = cache['f_all_norm']
    d_f_all = (d_f - f * np.dot(f, d_f)) / f_all_norm
    
    # Split gradient: [tensor_tokens | direct_features]
    d_tensor = d_f_all[:n_tensor_tokens]
    d_direct = d_f_all[n_tensor_tokens:]
    
    # --- Tensor product backward ---
    # tensor_tokens = a * b, where a = tp_proj_a @ f1, b = tp_proj_b @ f2
    a, b = cache['tp_a'], cache['tp_b']
    f1, f2 = cache['f1'], cache['f2']
    
    d_a = d_tensor * b
    d_b = d_tensor * a
    
    # tp_proj_a, tp_proj_b gradients
    G['tp_proj_a'] += np.outer(d_a, f1)
    G['tp_proj_b'] += np.outer(d_b, f2)
    
    # Gradient to f1, f2 from tensor product
    d_f1_from_tp = P['tp_proj_a'].T @ d_a
    d_f2_from_tp = P['tp_proj_b'].T @ d_b
    
    # Through f1 normalization: f1 = f1_raw / ||f1_raw||
    f1_raw, f1_norm = cache['f1_raw'], cache['f1_norm']
    f1 = cache['f1']
    d_f1_raw = (d_f1_from_tp - f1 * np.dot(f1, d_f1_from_tp)) / f1_norm
    
    d_l1x = d_f1_raw * cache['r1x']
    d_r1x = d_f1_raw * cache['l1x']
    G['bae1_l'] += np.outer(d_l1x, x_norm)
    G['bae1_r'] += np.outer(d_r1x, x_norm)
    
    # Through f2 normalization
    f2_raw, f2_norm = cache['f2_raw'], cache['f2_norm']
    f2 = cache['f2']
    d_f2_raw = (d_f2_from_tp - f2 * np.dot(f2, d_f2_from_tp)) / f2_norm
    
    d_l2x = d_f2_raw * cache['r2x']
    d_r2x = d_f2_raw * cache['l2x']
    G['bae2_l'] += np.outer(d_l2x, x_norm)
    G['bae2_r'] += np.outer(d_r2x, x_norm)
    
    # --- Direct multi-degree backward ---
    offset = 0
    for deg_idx, degree in enumerate(bae_degrees):
        n_tok = tokens_per_degree[deg_idx]
        d_f_degree = d_direct[offset:offset + n_tok]
        
        projections = [cache[f'proj_deg{degree}_{d}'] for d in range(degree)]
        
        for d in range(degree):
            other_prods = np.ones(n_tok)
            for other_d in range(degree):
                if other_d != d:
                    other_prods *= projections[other_d]
            
            d_proj = d_f_degree * other_prods
            G[f'bae_deg{degree}_{d}'] += np.outer(d_proj, x_norm)
        
        offset += n_tok
    
    # =========================================================================
    # Gradient Clipping
    # =========================================================================
    bae_max_norm = 1.0
    for key in P:
        if 'bae' in key or 'tp_' in key:
            gn = np.linalg.norm(G[key])
            if gn > bae_max_norm:
                G[key] *= bae_max_norm / (gn + 1e-8)
    
    max_grad_norm = 5.0
    total_norm = np.sqrt(sum(np.sum(G[k] ** 2) for k in G))
    if total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + 1e-8)
        for k in G:
            G[k] *= scale
    
    return text_loss, recon_loss, total_loss

# =============================================================================
# Inference
# =============================================================================
def inference(idx):
    x_raw = train_images_small[idx].copy()
    x_norm = x_raw / (np.sqrt(np.sum(x_raw ** 2)) + 1e-8)
    
    # BAE × BAE encode
    cache = {}
    f = bae_encode(x_norm, cache)
    
    # Vision projection (bilinear)
    vl = P['vis_proj_l'] @ f
    vr = P['vis_proj_r'] @ f
    vis_emb_base = vl * vr
    
    all_kv = []
    half = head_dim // 2
    
    for pos in range(n_vis_tokens):
        x = vis_emb_base + P['wpe'][pos]
        x, _ = rmsnorm(x)
        
        for li in range(n_layer):
            x_res = x
            x_ln, _ = rmsnorm(x)
            q = P[f'L{li}.wq'] @ x_ln
            k = P[f'L{li}.wk'] @ x_ln
            v = P[f'L{li}.wv'] @ x_ln
            all_kv.append((k, v))
            
            x_attn = np.zeros(n_embd)
            for h in range(n_head):
                hs, he = h * head_dim, (h + 1) * head_dim
                hm = hs + half
                q1, q2 = q[hs:hm], q[hm:he]
                scores = np.zeros(len(all_kv))
                for t in range(len(all_kv)):
                    s1 = all_kv[t][0][hs:hm] @ q1
                    s2 = all_kv[t][0][hm:he] @ q2
                    scores[t] = (s1 * s2) / head_dim
                w = softmax(scores)
                for t in range(len(all_kv)):
                    x_attn[hs:he] += w[t] * all_kv[t][1][hs:he]
            
            x = x_res + P[f'L{li}.wo'] @ x_attn
            x_res = x
            x_ln, _ = rmsnorm(x)
            mlp_hidden = np.maximum(0, P[f'L{li}.mlp_fc1'] @ x_ln)
            x = x_res + P[f'L{li}.mlp_fc2'] @ mlp_hidden
    
    token_id = BOS
    generated = []
    for t in range(max_text_len):
        pos = n_vis_tokens + t
        x = P['wte'][token_id] + P['wpe'][pos]
        x, _ = rmsnorm(x)
        
        for li in range(n_layer):
            x_res = x
            x_ln, _ = rmsnorm(x)
            q = P[f'L{li}.wq'] @ x_ln
            k = P[f'L{li}.wk'] @ x_ln
            v = P[f'L{li}.wv'] @ x_ln
            all_kv.append((k, v))
            
            x_attn = np.zeros(n_embd)
            for h in range(n_head):
                hs, he = h * head_dim, (h + 1) * head_dim
                hm = hs + half
                q1, q2 = q[hs:hm], q[hm:he]
                scores = np.zeros(len(all_kv))
                for t2 in range(len(all_kv)):
                    s1 = all_kv[t2][0][hs:hm] @ q1
                    s2 = all_kv[t2][0][hm:he] @ q2
                    scores[t2] = (s1 * s2) / head_dim
                w = softmax(scores)
                for t2 in range(len(all_kv)):
                    x_attn[hs:he] += w[t2] * all_kv[t2][1][hs:he]
            
            x = x_res + P[f'L{li}.wo'] @ x_attn
            x_res = x
            x_ln, _ = rmsnorm(x)
            mlp_hidden = np.maximum(0, P[f'L{li}.mlp_fc1'] @ x_ln)
            x = x_res + P[f'L{li}.mlp_fc2'] @ mlp_hidden
        
        logits = P['lm_head'] @ x
        token_id = int(np.argmax(logits))
        if token_id in (EOS, BOS):
            break
        generated.append(text_chars[token_id])
    
    return ''.join(generated)

# =============================================================================
# Adam Optimizer
# =============================================================================
learning_rate = 0.003
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
M = {k: np.zeros_like(v) for k, v in P.items()}
V_adam = {k: np.zeros_like(v) for k, v in P.items()}

def adam_step(step, lr):
    for k in P:
        M[k] = beta1 * M[k] + (1 - beta1) * G[k]
        V_adam[k] = beta2 * V_adam[k] + (1 - beta2) * (G[k] ** 2)
        m_hat = M[k] / (1 - beta1 ** (step + 1))
        v_hat = V_adam[k] / (1 - beta2 ** (step + 1))
        P[k] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

# =============================================================================
# Training Loop
# =============================================================================
alpha_recon = 0.01
num_steps = 7000
print(f"\n--- training ({num_steps} steps, BAE×BAE + Tensor Attention) ---")
t0 = time.time()

for step in range(num_steps):
    idx = step % len(train_images_small)
    label = train_labels[idx]
    name = digit_names[label]
    text_tokens = [BOS] + [char_to_id[ch] for ch in name] + [EOS]
    
    text_loss, recon_loss, total_loss = forward_backward(idx, text_tokens, alpha_recon)
    
    lr_t = learning_rate * (1 - step / num_steps)
    adam_step(step, lr_t)
    
    unstable = False
    for k in P:
        if np.any(np.isnan(P[k])) or np.any(np.isinf(P[k])):
            print(f"NaN/Inf detected in {k} at step {step}")
            unstable = True
            break
    if unstable:
        break
    
    if (step + 1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"step {step+1:5d}/{num_steps} | loss {total_loss:.4f} | "
              f"text {text_loss:.4f} | recon {recon_loss:.4f} | "
              f"target: {name} | {elapsed:.1f}s")

print(f"\ntotal training time: {time.time() - t0:.1f}s")

# =============================================================================
# Evaluation
# =============================================================================
print(f"\n--- inference: BAE×BAE + Tensor Product Attention ---")
correct = 0
total = 20

for sample_idx in range(total):
    idx = len(train_images_small) - 1 - sample_idx * 100
    label = train_labels[idx]
    true_name = digit_names[label]
    
    pred_name = inference(idx)
    ok = (pred_name == true_name)
    correct += ok
    mark = '✓' if ok else '✗'
    print(f"  [{label}] true: {true_name:>5s} | generated: {pred_name:<8s} {mark}")

print(f"\naccuracy: {correct}/{total} = {100*correct/total:.0f}%")
