"""
Pure VLM: See a digit, say its name.
Bilinear AE visual tokenizer + GPT in pure Python, zero dependencies.

Input:  MNIST 28x28 grayscale image
Output: generated text like "seven"

Architecture:
  Image pixels -> [BAE encode] -> vision tokens -> [GPT] -> text tokens
  
  Training: teacher-forced on sequences like:
    [VIS_0] [VIS_1] ... [VIS_7] [BOS] s e v e n [EOS]
    |--- vision tokens ---|  |--- text tokens ---|

  Inference: feed vision tokens, then autoregressively generate text.

This file is the complete algorithm. Everything else is just efficiency.
"""

import os
import math
import random
import struct
import gzip
random.seed(42)

# =============================================================================
# Autograd
# =============================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data + 1e-10), (self,), (1/(self.data + 1e-10),))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# =============================================================================
# Primitives
# =============================================================================
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# =============================================================================
# MNIST loader (pure Python, no dependencies)
# =============================================================================
def load_mnist(images_path, labels_path):
    """Load MNIST from raw gzipped IDX files."""
    with gzip.open(labels_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = list(f.read(n))
    with gzip.open(images_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        pixels = f.read(n * rows * cols)
        images = []
        for i in range(n):
            offset = i * rows * cols
            img = [pixels[offset + j] / 255.0 for j in range(rows * cols)]
            images.append(img)
    return images, labels

def download_mnist():
    """Download MNIST if not present."""
    import urllib.request
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]
    os.makedirs('mnist', exist_ok=True)
    for fname in files:
        path = os.path.join('mnist', fname)
        if not os.path.exists(path):
            print(f"downloading {fname}...")
            urllib.request.urlretrieve(base + fname, path)

# download_mnist()
train_images, train_labels = load_mnist('mnist/train-images-idx3-ubyte.gz', 'mnist/train-labels-idx1-ubyte.gz')
print(f"loaded {len(train_images)} training images, 28x28={len(train_images[0])}px")

# =============================================================================
# Tokenizer: text vocabulary for digit names
# =============================================================================
digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
text_chars = sorted(set(''.join(digit_names)))  # unique characters in digit names
BOS = len(text_chars)      # beginning of text sequence
EOS = len(text_chars) + 1  # end of text sequence
text_vocab_size = len(text_chars) + 2  # chars + BOS + EOS
print(f"text vocab: {text_chars} + BOS + EOS = {text_vocab_size} tokens")

def char_to_id(ch):
    return text_chars.index(ch)

def id_to_char(tid):
    if tid == BOS: return '<BOS>'
    if tid == EOS: return '<EOS>'
    return text_chars[tid]

# =============================================================================
# Downscale 28x28 -> 8x8 via average pooling (pure Python)
# =============================================================================
IMG_ORIG = 28
IMG_SIZE = 8  # downscaled size. 8x8 = 64 pixels, manageable for Value autograd
N_PIXELS = IMG_SIZE * IMG_SIZE

def downscale(img_28x28):
    """Average-pool 28x28 -> 8x8. Handle 28 not divisible by 8 with overlapping bins."""
    out = [0.0] * N_PIXELS
    for r in range(IMG_SIZE):
        for c in range(IMG_SIZE):
            r0 = int(r * IMG_ORIG / IMG_SIZE)
            r1 = int((r + 1) * IMG_ORIG / IMG_SIZE)
            c0 = int(c * IMG_ORIG / IMG_SIZE)
            c1 = int((c + 1) * IMG_ORIG / IMG_SIZE)
            total = 0.0
            count = 0
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    total += img_28x28[rr * IMG_ORIG + cc]
                    count += 1
            out[r * IMG_SIZE + c] = total / count if count > 0 else 0.0
    return out

# Preprocess all training images
print("downscaling images to 8x8...")
train_images_small = [downscale(img) for img in train_images]

# =============================================================================
# Model hyperparameters
# =============================================================================
n_layer = 1
n_embd = 16
n_head = 4
head_dim = n_embd // n_head
n_vis_tokens = 8   # number of vision tokens from BAE
max_text_len = 8   # max text length ("eight" = 5 chars + BOS + EOS = 7, fits in 8)
block_size = n_vis_tokens + max_text_len  # total sequence length

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# =============================================================================
# Parameters
# =============================================================================
state_dict = {}

# BAE encoder
state_dict['bae_l'] = matrix(n_vis_tokens, N_PIXELS, std=0.05)
state_dict['bae_r'] = matrix(n_vis_tokens, N_PIXELS, std=0.05)

# Vision: latent -> embedding
state_dict['vis_proj'] = matrix(n_embd, n_vis_tokens, std=0.08)

# Text: token embedding + shared output head
state_dict['wte'] = matrix(text_vocab_size, n_embd)

# Position embeddings (shared for full sequence)
state_dict['wpe'] = matrix(block_size, n_embd)

# Transformer layers
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Text output head (tied with wte conceptually, but separate for clarity)
state_dict['lm_head'] = matrix(text_vocab_size, n_embd)

# BAE reconstruction (kernel trick)
# Uses bae_l and bae_r, no extra params needed

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
print(f"sequence: {n_vis_tokens} vision + {max_text_len} text = {block_size} total")

# =============================================================================
# BAE encode + reconstruction loss
# =============================================================================
def input_normalize(x):
    norm_sq = sum(xi * xi for xi in x)
    scale = (norm_sq + 1e-8) ** -0.5
    return [xi * scale for xi in x]

def bae_encode(x):
    lx = linear(x, state_dict['bae_l'])
    rx = linear(x, state_dict['bae_r'])
    return [lx[j] * rx[j] for j in range(n_vis_tokens)]

def bae_recon_loss(x, f):
    L = state_dict['bae_l']
    R = state_dict['bae_r']
    recons = Value(0)
    for i in range(n_vis_tokens):
        for j in range(n_vis_tokens):
            ll = sum(L[i][k] * L[j][k] for k in range(N_PIXELS))
            rr = sum(R[i][k] * R[j][k] for k in range(N_PIXELS))
            recons = recons + f[i] * (ll * rr) * f[j]
    cross = sum(fi * fi for fi in f)
    xtx = sum(xi * xi for xi in x)
    return recons - Value(2) * cross + xtx

# =============================================================================
# Vision tokens -> embeddings
# =============================================================================
def vision_to_embeddings(f):
    """Convert BAE features to a list of n_vis_tokens embedding vectors."""
    base = linear(f, state_dict['vis_proj'])  # n_embd
    tokens = []
    for pos in range(n_vis_tokens):
        tok = [base[d] + state_dict['wpe'][pos][d] for d in range(n_embd)]
        tokens.append(tok)
    return tokens

# =============================================================================
# GPT: processes the full sequence (vision + text) causally
# =============================================================================
def gpt_forward(token_id, pos_id, keys, values, embedding=None):
    """
    Single-step GPT forward.
    If embedding is provided, use it directly (for vision tokens).
    If token_id is provided, look up wte (for text tokens).
    """
    if embedding is not None:
        x = embedding  # already includes position from vision_to_embeddings
    else:
        tok_emb = state_dict['wte'][token_id]
        pos_emb = state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
    
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# =============================================================================
# Training
# =============================================================================
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)
alpha_recon = 0.01  # reconstruction loss weight

num_steps = 3000
print(f"\n--- training ({num_steps} steps) ---")

for step in range(num_steps):
    idx = step % len(train_images_small)
    img = train_images_small[idx]
    label = train_labels[idx]
    name = digit_names[label]
    
    # Build target text sequence: BOS + "seven" + EOS
    text_tokens = [BOS] + [char_to_id(ch) for ch in name] + [EOS]
    
    # 1. BAE encode image
    x_pixels = [Value(p) for p in img]
    x_norm = input_normalize(x_pixels)
    f = bae_encode(x_norm)
    
    # 2. Reconstruction loss
    recon_loss = bae_recon_loss(x_norm, f)
    
    # 3. Vision token embeddings
    vis_embs = vision_to_embeddings(f)
    
    # 4. Forward full sequence through GPT, collecting text prediction losses
    #    Sequence: [vis_0, vis_1, ..., vis_7, BOS, t, h, r, e, e, EOS]
    #    We predict text tokens only (starting after vision tokens + BOS)
    keys = [[] for _ in range(n_layer)]
    values_kv = [[] for _ in range(n_layer)]
    text_losses = []
    
    # Feed vision tokens (no prediction loss, just build context)
    for pos in range(n_vis_tokens):
        _ = gpt_forward(None, pos, keys, values_kv, embedding=vis_embs[pos])
    
    # Feed text tokens with teacher forcing
    n_text = len(text_tokens) - 1  # predict next for each except last
    for t in range(n_text):
        pos = n_vis_tokens + t
        logits = gpt_forward(text_tokens[t], pos, keys, values_kv)
        probs = softmax(logits)
        target_id = text_tokens[t + 1]
        text_losses.append(-probs[target_id].log())
    
    text_loss = sum(text_losses) / len(text_losses)
    
    # 5. Joint loss
    loss = text_loss + alpha_recon * recon_loss
    
    # Backward
    loss.backward()
    
    # Adam
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    
    if (step + 1) % 100 == 0:
        print(f"step {step+1:5d}/{num_steps} | loss {loss.data:.4f} | text {text_loss.data:.4f} | recon {recon_loss.data:.4f} | target: {name}")

# =============================================================================
# Inference: show image, generate text
# =============================================================================
print("\n--- inference: seeing digits, saying names ---")

for sample_idx in range(20):
    # Pick a random test-ish sample (from end of training set)
    idx = len(train_images_small) - 1 - sample_idx * 100
    img = train_images_small[idx]
    label = train_labels[idx]
    true_name = digit_names[label]
    
    # Encode image
    x_pixels = [Value(p) for p in img]
    x_norm = input_normalize(x_pixels)
    f = bae_encode(x_norm)
    vis_embs = vision_to_embeddings(f)
    
    # Feed vision tokens
    keys = [[] for _ in range(n_layer)]
    values_kv = [[] for _ in range(n_layer)]
    for pos in range(n_vis_tokens):
        _ = gpt_forward(None, pos, keys, values_kv, embedding=vis_embs[pos])
    
    # Generate text autoregressively
    token_id = BOS
    generated = []
    for t in range(max_text_len):
        pos = n_vis_tokens + t
        logits = gpt_forward(token_id, pos, keys, values_kv)
        probs = softmax(logits)
        # Greedy decoding
        token_id = max(range(text_vocab_size), key=lambda c: probs[c].data)
        if token_id == EOS:
            break
        if token_id == BOS:
            break
        generated.append(text_chars[token_id])
    
    pred_name = ''.join(generated)
    mark = '✓' if pred_name == true_name else '✗'
    print(f"  [{label}] true: {true_name:>5s} | generated: {pred_name:<8s} {mark}")
