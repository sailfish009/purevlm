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
