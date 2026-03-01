# The most atomic Vision-Language Model, in pure Python

Building on @karpathy's pure GPT, this adds vision to the same minimalist framework. A complete VLM — see a digit, say its name — in dependency-free Python.

## What it does

Feed it an MNIST digit. It generates the word:

```
[8] true: eight | generated: eight    ✓
[3] true: three | generated: three    ✓
[7] true: seven | generated: seven    ✓
[0] true:  zero | generated: zero     ✓
```

**95% accuracy** on MNIST. ~32K parameters. No GPU. Pure NumPy.

## How it works

The architecture follows a single mathematical principle end-to-end:

```
Image pixels → [BAE × BAE] → vision tokens → [Tensor Attention GPT] → text tokens
```

During training, the model sees sequences like:

```
[VIS_0] [VIS_1] ... [VIS_27] [BOS] s e v e n [EOS]
```

During inference, it encodes the image into vision tokens, then autoregressively generates text — exactly like a language model, but the prompt is an image.

## The visual tokenizer: BAE × BAE

The original Bilinear Autoencoder (Lieberum et al., 2025) captures second-order pixel interactions:

```
f_j = dot(l_j, x) * dot(r_j, x)     — degree-2 polynomial kernel
```

Two dot products — that's it. This captures pairwise pixel interactions without ever materializing the quadratic space x⊗x.

**BAE × BAE** takes this further. Two independent BAEs each produce stable, normalized degree-2 features. Their tensor product reaches degree-4 — capturing **three-way and four-way pixel interactions** — while each factor remains independently well-conditioned:

```
f₁ = (L₁x) ⊙ (R₁x)          — BAE₁: degree-2
f₂ = (L₂x) ⊙ (R₂x)          — BAE₂: degree-2
token_j = (Pₐ[j]·f₁)(Pᵦ[j]·f₂)  — degree-4 via tensor product
```

This is the kernel trick applied twice: the model implicitly operates in a ~1.25M-dimensional feature space (for 14×14 images) while only computing with vectors of length 10.

Direct degree-2 and degree-3 features flow alongside as skip connections, giving the transformer access to multiple levels of polynomial interaction simultaneously.

## Tensor Product Attention

Standard attention computes similarity as a single dot product — a degree-2 kernel:

```
score(q, k) = qᵀk / √d
```

We replace this with **tensor product attention**, splitting each head in half:

```
score(q, k) = (q₁ᵀk₁)(q₂ᵀk₂) / d     — degree-4 kernel
```

This asks not just "are these tokens similar?" but **"does feature-group-A match AND feature-group-B match?"** — a conjunctive condition that distinguishes digits like 8 vs 0, where the top half is similar but the bottom half differs.

**Zero extra parameters.** Same Q, K, V matrices, just a different score function. This single change improved accuracy from 85% to 95%.

## Everything else is the same GPT

The transformer is: RMSNorm, multi-head causal attention, ReLU MLP, Adam optimizer. Vision tokens and text tokens flow through the same attention layers in a single unified sequence.

## The kernel trick runs the whole show

Every component speaks the same mathematical language:

| Component | Operation | Kernel degree |
|-----------|-----------|:---:|
| BAE visual features | `(Lx) ⊙ (Rx)` | 2 |
| Multi-degree skip | `(L₁x)(L₂x)(L₃x)` | 3 |
| BAE × BAE tokens | `(Pₐ·f₁)(Pᵦ·f₂)` | 4 |
| Vision projection | `(Wₗ·f) ⊙ (Wᵣ·f)` | 2 |
| Tensor attention | `(q₁·k₁)(q₂·k₂)` | 4 |

No component was designed in isolation. Each is a polynomial kernel operating at a different degree, and they compose naturally because **products of kernels are kernels**.

## The point

This isn't meant to compete with Qwen-VL or LLaVA. It's meant to explore whether the kernel trick — the same mathematical insight from SVMs — can serve as a unified foundation for vision-language models, replacing the engineering stack of ViT + adapter + LLM with a single coherent algebraic structure.

Every gradient is traceable. Every matrix multiply is visible. The entire model fits in one file.

As the original code says: *"This file is the complete algorithm. Everything else is just efficiency."*

That now includes vision — and a hint at what a kernel-native VLM architecture could become.
