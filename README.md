# The most atomic Vision-Language Model, in pure Python

Building on [@karpathy's pure GPT](https://gist.github.com/karpathy/), this adds vision to the same minimalist framework. A complete VLM — see a digit, say its name — in ~250 lines of dependency-free Python.

## What it does

Feed it an MNIST digit. It generates the word:

```
[8] true: eight | generated: eight    ✓
[3] true: three | generated: three    ✓
[7] true: seven | generated: seven    ✓
[0] true:  zero | generated: zero     ✓
```

80% accuracy on MNIST. 5,024 parameters. No GPU. No NumPy. No PyTorch.

## How it works

The architecture follows a simple pipeline:

```
Image pixels → [Bilinear AE] → vision tokens → [GPT] → text tokens
```

During training, the model sees sequences like:

```
[VIS_0] [VIS_1] ... [VIS_7] [BOS] s e v e n [EOS]
```

During inference, it encodes the image into vision tokens, then autoregressively generates text — exactly like a language model, but the prompt is an image.

### The visual tokenizer: Bilinear Autoencoder

Instead of ViT-style patch embeddings, we use a Bilinear Autoencoder ([Lieberum et al., 2025](https://arxiv.org/abs/2510.16820)) as the visual tokenizer. Each latent captures a second-order interaction:

```python
f_j = dot(l_j, x) * dot(r_j, x)
```

Two dot products — that's it. This captures pairwise pixel interactions without ever materializing the quadratic space `x⊗x`. The kernel trick keeps reconstruction tractable:

```
BB⊤ = (LL⊤) ⊙ (RR⊤)
```

### Everything else is the same GPT

The transformer is identical to Karpathy's original: scalar-level autograd, RMSNorm, multi-head causal attention, ReLU MLP, Adam optimizer. Vision tokens and text tokens flow through the same attention layers in a single unified sequence.

## The point

This isn't meant to compete with Qwen-VL or LLaVA. It's meant to make the VLM algorithm *transparent*. Every gradient is traceable. Every matrix multiply is a nested loop. You can read the entire model in one sitting.

As the original code says: **"This file is the complete algorithm. Everything else is just efficiency."**

That now includes vision.

--

```
loaded 60000 training images, 28x28=784px
text vocab: ['e', 'f', 'g', 'h', 'i', 'n', 'o', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z'] + BOS + EOS = 17 tokens
downscaling images to 8x8...
num params: 5024
sequence: 8 vision + 8 text = 16 total

--- training (3000 steps) ---
step   100/3000 | loss 1.0359 | text 1.0261 | recon 0.9877 | target: one
step   200/3000 | loss 0.2949 | text 0.2846 | recon 1.0282 | target: two
step   300/3000 | loss 1.1154 | text 1.1054 | recon 1.0024 | target: six
step   400/3000 | loss 0.6950 | text 0.6826 | recon 1.2359 | target: zero
step   500/3000 | loss 0.6267 | text 0.6142 | recon 1.2523 | target: eight
step   600/3000 | loss 0.4839 | text 0.4559 | recon 2.8045 | target: seven
step   700/3000 | loss 0.4538 | text 0.4404 | recon 1.3316 | target: five
step   800/3000 | loss 0.3074 | text 0.2960 | recon 1.1380 | target: five
step   900/3000 | loss 0.3001 | text 0.2898 | recon 1.0250 | target: five
step  1000/3000 | loss 0.2857 | text 0.2756 | recon 1.0136 | target: six
step  1100/3000 | loss 0.1405 | text 0.1246 | recon 1.5981 | target: four
step  1200/3000 | loss 0.3140 | text 0.2997 | recon 1.4269 | target: two
step  1300/3000 | loss 1.4336 | text 1.4226 | recon 1.1023 | target: two
step  1400/3000 | loss 0.1257 | text 0.1077 | recon 1.8040 | target: nine
step  1500/3000 | loss 0.2428 | text 0.2231 | recon 1.9722 | target: seven
step  1600/3000 | loss 0.4303 | text 0.4038 | recon 2.6475 | target: three
step  1700/3000 | loss 0.0422 | text 0.0290 | recon 1.3139 | target: one
step  1800/3000 | loss 0.3149 | text 0.3015 | recon 1.3375 | target: three
step  1900/3000 | loss 0.3520 | text 0.3337 | recon 1.8216 | target: seven
step  2000/3000 | loss 0.0605 | text 0.0459 | recon 1.4641 | target: zero
step  2100/3000 | loss 0.1452 | text 0.1302 | recon 1.4988 | target: nine
step  2200/3000 | loss 0.1642 | text 0.1482 | recon 1.6069 | target: nine
step  2300/3000 | loss 0.1622 | text 0.1459 | recon 1.6249 | target: nine
step  2400/3000 | loss 0.0283 | text 0.0025 | recon 2.5780 | target: two
step  2500/3000 | loss 0.0331 | text 0.0154 | recon 1.7613 | target: two
step  2600/3000 | loss 0.0151 | text 0.0002 | recon 1.4919 | target: one
step  2700/3000 | loss 0.0709 | text 0.0587 | recon 1.2157 | target: eight
step  2800/3000 | loss 0.0438 | text 0.0244 | recon 1.9369 | target: four
step  2900/3000 | loss 0.0226 | text 0.0098 | recon 1.2872 | target: zero
step  3000/3000 | loss 0.0831 | text 0.0703 | recon 1.2755 | target: five

--- inference: seeing digits, saying names ---
  [8] true: eight | generated: eight    ✓
  [9] true:  nine | generated: nine     ✓
  [4] true:  four | generated: four     ✓
  [3] true: three | generated: three    ✓
  [4] true:  four | generated: four     ✓
  [2] true:   two | generated: two      ✓
  [8] true: eight | generated: three    ✗
  [3] true: three | generated: eight    ✗
  [1] true:   one | generated: one      ✓
  [0] true:  zero | generated: zero     ✓
  [4] true:  four | generated: four     ✓
  [7] true: seven | generated: seven    ✓
  [5] true:  five | generated: five     ✓
  [9] true:  nine | generated: nine     ✓
  [0] true:  zero | generated: seven    ✗
  [9] true:  nine | generated: nine     ✓
  [3] true: three | generated: three    ✓
  [7] true: seven | generated: nine     ✗
  [8] true: eight | generated: eight    ✓
  [7] true: seven | generated: seven    ✓
```
