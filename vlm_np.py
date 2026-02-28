"""
Pure VLM (NumPy): numpy version
BAE with stabilized reconstruction loss + feature normalization.
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
IMG_SIZE = 8
N_PIXELS = IMG_SIZE * IMG_SIZE

def downscale_all(images, orig=28, target=8):
    n = len(images)
    imgs = images.reshape(n, orig, orig)
    out = np.zeros((n, target, target))
    for r in range(target):
        for c in range(target):
            r0, r1 = int(r * orig / target), int((r+1) * orig / target)
            c0, c1 = int(c * orig / target), int((c+1) * orig / target)
            out[:, r, c] = imgs[:, r0:r1, c0:c1].mean(axis=(1, 2))
    return out.reshape(n, target * target)

print("downscaling images to 8x8...")
train_images_small = downscale_all(train_images)

# =============================================================================
# Hyperparameters
# =============================================================================
n_layer = 1
n_embd = 24 #  16
n_head = 4
head_dim = n_embd // n_head
n_vis_tokens = 8
max_text_len = 8
block_size = n_vis_tokens + max_text_len

# =============================================================================
# Parameters & Gradients
# =============================================================================
def param(shape, std=0.08):
    return np.random.randn(*shape).astype(np.float64) * std

P = {}
G = {}

# BAE
P['bae_l'] = param((n_vis_tokens, N_PIXELS), 0.05)
P['bae_r'] = param((n_vis_tokens, N_PIXELS), 0.05)

# Vision projection
P['vis_proj'] = param((n_embd, n_vis_tokens))

# Embeddings
P['wte'] = param((text_vocab_size, n_embd))
P['wpe'] = param((block_size, n_embd))

# Transformer
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
print(f"num params: {n_params}")
print(f"architecture: standard GPT with BAE visual tokenizer")
print(f"sequence: {n_vis_tokens} vision + {max_text_len} text = {block_size} total")

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
# Forward + Backward (STABILIZED BAE)
# =============================================================================
def forward_backward(img, text_tokens, alpha_recon=0.01):
    for k in G:
        G[k][:] = 0.0
    
    cache = {}
    seq_len = n_vis_tokens + len(text_tokens)
    
    # =========================================================================
    # FORWARD
    # =========================================================================
    
    # BAE Encode with feature normalization
    x_raw = img.copy()
    x_norm = x_raw / (np.sqrt(np.sum(x_raw ** 2)) + 1e-8)
    cache['x_norm'] = x_norm
    
    lx = P['bae_l'] @ x_norm
    rx = P['bae_r'] @ x_norm
    
    # CRITICAL: Normalize BAE features to prevent explosion
    f_raw = lx * rx
    f_norm = np.linalg.norm(f_raw) + 1e-8
    f = f_raw / f_norm  # Unit norm features
    cache['bae_lx'], cache['bae_rx'], cache['f'] = lx, rx, f
    cache['f_norm'] = f_norm
    
    # Reconstruction Loss (STABILIZED)
    L, R = P['bae_l'], P['bae_r']
    
    # Normalize L and R for stable kernel computation
    L_norm = np.linalg.norm(L, axis=1, keepdims=True) + 1e-8
    R_norm = np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    L_n = L / L_norm
    R_n = R / R_norm
    
    LL = L_n @ L_n.T
    RR = R_n @ R_n.T
    BB = LL * RR
    recon_loss = f @ BB @ f - 2 * (f @ f) + (x_norm @ x_norm)
    
    # Clip recon loss for monitoring
    recon_loss_clipped = np.clip(recon_loss, 0, 10.0)
    cache['BB'], cache['LL'], cache['RR'] = BB, LL, RR
    cache['L_norm'], cache['R_norm'] = L_norm, R_norm
    
    # Vision Projection
    vis_emb_base = P['vis_proj'] @ f
    cache['vis_emb_base'] = vis_emb_base
    
    # Build Embeddings
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
    cache['normed_embs'] = normed_embs
    cache['rms_scales_init'] = rms_scales_init
    
    # Transformer Forward
    for li in range(n_layer):
        x_seq = normed_embs.copy()
        
        # Pre-attention RMSNorm
        x_ln = np.zeros_like(x_seq)
        rms_scales_attn = np.zeros(seq_len)
        for i in range(seq_len):
            x_ln[i], rms_scales_attn[i] = rmsnorm(x_seq[i])
        cache[f'L{li}_x_ln'] = x_ln
        cache[f'L{li}_rms_attn'] = rms_scales_attn
        
        # Q, K, V
        Q = x_ln @ P[f'L{li}.wq'].T
        K = x_ln @ P[f'L{li}.wk'].T
        V = x_ln @ P[f'L{li}.wv'].T
        cache[f'L{li}_Q'] = Q
        cache[f'L{li}_K'] = K
        cache[f'L{li}_V'] = V
        
        # Multi-head Attention
        attn_out = np.zeros((seq_len, n_embd))
        attn_weights_cache = []
        
        for i in range(seq_len):
            head_outputs = []
            head_weights = []
            
            for h in range(n_head):
                hs, he = h * head_dim, (h + 1) * head_dim
                q_h = Q[i, hs:he]
                k_h = K[:i+1, hs:he]
                v_h = V[:i+1, hs:he]
                
                scores = (k_h @ q_h) / np.sqrt(head_dim)
                w = softmax(scores)
                head_weights.append(w)
                
                head_out = w @ v_h
                head_outputs.append(head_out)
            
            attn_out[i] = np.concatenate(head_outputs)
            attn_weights_cache.append(head_weights)
        
        cache[f'L{li}_attn_weights'] = attn_weights_cache
        
        attn_proj = attn_out @ P[f'L{li}.wo'].T
        x_after_attn = x_seq + attn_proj
        cache[f'L{li}_x_after_attn'] = x_after_attn
        
        # ReLU MLP
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
        cache[f'L{li}_output'] = normed_embs.copy()
    
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
    total_loss = text_loss + alpha_recon * recon_loss_clipped
    
    cache['normed_embs_final'] = normed_embs
    cache['all_probs'] = all_probs
    cache['n_text'] = n_text
    cache['seq_len'] = seq_len
    
    # =========================================================================
    # BACKWARD (STABILIZED)
    # =========================================================================
    
    # LM Head Backward
    d_final = np.zeros_like(normed_embs)
    
    for probs, target, pos in all_probs:
        d_logits = probs.copy()
        d_logits[target] -= 1.0
        d_logits /= n_text
        
        G['lm_head'] += np.outer(d_logits, normed_embs[pos])
        d_final[pos] += P['lm_head'].T @ d_logits
    
    # Transformer Backward
    d_x = d_final.copy()
    
    for li in range(n_layer - 1, -1, -1):
        # MLP Backward
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
        
        # Attention Backward
        d_attn_proj = d_x_after_attn.copy()
        attn_out_cache = cache.get(f'L{li}_attn_out', np.zeros_like(d_x))
        
        G[f'L{li}.wo'] += d_attn_proj.T @ attn_out_cache
        d_attn_out = d_attn_proj @ P[f'L{li}.wo']
        
        d_x_seq = d_x_after_attn.copy()
        
        Q = cache[f'L{li}_Q']
        K = cache[f'L{li}_K']
        V = cache[f'L{li}_V']
        all_weights = cache[f'L{li}_attn_weights']
        
        d_Q = np.zeros_like(Q)
        d_K = np.zeros_like(K)
        d_V = np.zeros_like(V)
        
        for i in range(seq_len):
            for h in range(n_head):
                hs, he = h * head_dim, (h + 1) * head_dim
                w = all_weights[i][h]
                d_head_out = d_attn_out[i, hs:he]
                v_h = V[:i+1, hs:he]
                
                d_w = v_h @ d_head_out
                for t in range(i + 1):
                    d_V[t, hs:he] += w[t] * d_head_out
                
                d_scores = w * (d_w - np.dot(w, d_w))
                
                q_h = Q[i, hs:he]
                for t in range(i + 1):
                    k_h = K[t, hs:he]
                    ds = d_scores[t] / np.sqrt(head_dim)
                    d_Q[i, hs:he] += ds * k_h
                    d_K[t, hs:he] += ds * q_h
        
        x_ln = cache[f'L{li}_x_ln']
        G[f'L{li}.wq'] += d_Q.T @ x_ln
        G[f'L{li}.wk'] += d_K.T @ x_ln
        G[f'L{li}.wv'] += d_V.T @ x_ln
        
        d_x_ln = d_Q @ P[f'L{li}.wq'] + d_K @ P[f'L{li}.wk'] + d_V @ P[f'L{li}.wv']
        
        rms_attn = cache[f'L{li}_rms_attn']
        for i in range(seq_len):
            d_x_seq[i] += d_x_ln[i] * rms_attn[i]
        
        d_x = d_x_seq
    
    # Initial RMSNorm Backward
    rms_init = cache['rms_scales_init']
    d_emb = np.zeros((seq_len, n_embd))
    for i in range(seq_len):
        d_emb[i] = d_x[i] * rms_init[i]
    
    # Embedding Backward
    d_vis_emb_base = np.zeros(n_embd)
    
    for pos in range(n_vis_tokens):
        d_vis_emb_base += d_emb[pos]
        G['wpe'][pos] += d_emb[pos]
    
    for t, tok in enumerate(text_tokens):
        pos = n_vis_tokens + t
        G['wte'][tok] += d_emb[pos]
        G['wpe'][pos] += d_emb[pos]
    
    # Vision Projection Backward
    f = cache['f']
    G['vis_proj'] += np.outer(d_vis_emb_base, f)
    d_f_proj = P['vis_proj'].T @ d_vis_emb_base
    
    # =============================================================================
    # BAE Backward (STABILIZED with feature normalization gradient)
    # =============================================================================
    BB = cache['BB']
    lx_val, rx_val = cache['bae_lx'], cache['bae_rx']
    x_norm = cache['x_norm']
    f_norm = cache['f_norm']
    
    # Gradient through normalized f (simplified - through f only)
    # d(recon)/df = 2*BB@f - 4*f
    d_f_recon = alpha_recon * (2 * BB @ f - 4 * f)
    
    # Total gradient for f
    d_f = d_f_proj + d_f_recon
    
    # Account for feature normalization: f = f_raw / ||f_raw||
    # d(f_raw)/d(f) = projection onto orthogonal complement
    f_raw = lx_val * rx_val
    d_f_raw = (d_f - f * np.dot(f, d_f)) / f_norm
    
    # f_raw = lx * rx (element-wise)
    d_lx = d_f_raw * rx_val
    d_rx = d_f_raw * lx_val
    
    # Gradient for L, R
    G['bae_l'] += np.outer(d_lx, x_norm)
    G['bae_r'] += np.outer(d_rx, x_norm)
    
    # =============================================================================
    # Aggressive Gradient Clipping (CRITICAL for BAE stability)
    # =============================================================================
    # Clip BAE gradients separately
    bae_max_norm = 1.0
    for key in ['bae_l', 'bae_r']:
        grad_norm = np.linalg.norm(G[key])
        if grad_norm > bae_max_norm:
            G[key] *= bae_max_norm / (grad_norm + 1e-8)
    
    # Global gradient clipping
    max_grad_norm = 5.0
    total_norm = 0.0
    for k in G:
        total_norm += np.sum(G[k] ** 2)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + 1e-8)
        for k in G:
            G[k] *= scale
    
    return text_loss, recon_loss_clipped, total_loss

# =============================================================================
# Inference
# =============================================================================
def inference(img):
    x_norm = img / (np.sqrt(np.sum(img ** 2)) + 1e-8)
    
    lx = P['bae_l'] @ x_norm
    rx = P['bae_r'] @ x_norm
    f_raw = lx * rx
    f_norm = np.linalg.norm(f_raw) + 1e-8
    f = f_raw / f_norm
    
    vis_emb_base = P['vis_proj'] @ f
    
    all_kv = []
    
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
                scores = np.zeros(len(all_kv))
                for t in range(len(all_kv)):
                    scores[t] = (all_kv[t][0][hs:he] @ q[hs:he]) / np.sqrt(head_dim)
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
                scores = np.zeros(len(all_kv))
                for t2 in range(len(all_kv)):
                    scores[t2] = (all_kv[t2][0][hs:he] @ q[hs:he]) / np.sqrt(head_dim)
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
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
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
num_steps = 5000 # 3000
print(f"\n--- training ({num_steps} steps, stabilized BAE) ---")
t0 = time.time()

for step in range(num_steps):
    idx = step % len(train_images_small)
    img = train_images_small[idx]
    label = train_labels[idx]
    name = digit_names[label]
    text_tokens = [BOS] + [char_to_id[ch] for ch in name] + [EOS]
    
    text_loss, recon_loss, total_loss = forward_backward(img, text_tokens, alpha_recon)
    
    lr_t = learning_rate * (1 - step / num_steps)
    adam_step(step, lr_t)
    
    # NaN/Inf detection
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
# Inference Evaluation
# =============================================================================
print("\n--- inference: seeing digits, saying names ---")
correct = 0
total = 20

for sample_idx in range(total):
    idx = len(train_images_small) - 1 - sample_idx * 100
    img = train_images_small[idx]
    label = train_labels[idx]
    true_name = digit_names[label]
    
    pred_name = inference(img)
    ok = (pred_name == true_name)
    correct += ok
    mark = '✓' if ok else '✗'
    print(f"  [{label}] true: {true_name:>5s} | generated: {pred_name:<8s} {mark}")

print(f"\naccuracy: {correct}/{total} = {100*correct/total:.0f}%")
