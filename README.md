## Sigma-Weighted Shuffle

A lightweight [ComfyUI](https://github.com/comfyanonymous/ComfyUI) model patch that locally permutes K/V tokens inside attention, with a sigma-aware schedule, entropy-gated strength, and a KL-based safety check, to diversify focus without destabilizing generation. 

## Short Math

All operations are built upon the attention maps. With input qkv and sigma, calculate the sigma weight and attention entropy. Using high entropy as a guide of when to shuffle and calculate kl divergence, output tuned qkv. When entropy is low, output original qkv. Since it's linear algebra, there's virtually no overhead.

## Math

Let σ ∈ [σₘᵢₙ, σₘₐₓ].
**Progress & weight.**
u = clamp( (ln σ − ln σₘᵢₙ) / (ln σₘₐₓ − ln σₘᵢₙ), 0, 1 ), 
w = sigmoid(α(u − ½)) / sigmoid(α/2).

**Temperature shaping.** With τ = 1 + (1 − u) and s = τ^(−1/2): q ← s·q, k ← s·k.

**Baseline attention.** A₀ = softmax(q kᵀ / √d).

**Local shuffle (per step).** Build a permutation P via a Sinkhorn-normalized, locality-biased cost on the token grid; apply to keys/values:
k′ = P k, v′ = P v.

**Entropy-gated mix.** Let H(A₀) be attention entropy and a ∝ intensity · H(A₀) · (1 + cos π(u − ½)) / 2.
Mix with small K bias:
kₘᵢₓ = (1 − λₖ)k + λₖ k′, vₘᵢₓ = (1 − λᵥ)v + λᵥ v′, with λₖ ≪ λᵥ ≤ a.

**KL safety.** Let A₁ = softmax(q kₘᵢₓᵀ / √d).
Downscale by 1 / (1 + λₖₗ · Dₖₗ(A₁ ∥ A₀)) and softly fade for late steps. 

## Installation

1. **Clone the node**

	Go to the custom_nodes directory.
	Run this command.
	```bash
	git clone https://github.com/geltz/sws
	```
3. **Restart ComfyUI.**

5. **Use the node**

   Add **“Sigma-Weighted Shuffle”** (category: `model_patches`) before your sampler; connect the `MODEL` input and set `intensity` (0–1).

   Recommended start: `intensity = 0.5`.

## Results

https://imgsli.com/NDIyMTcz

https://imgsli.com/NDIyMTc0

https://imgsli.com/NDIyMTc1
