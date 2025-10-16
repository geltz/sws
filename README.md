# Sigma-Weighted Shuffle

SWS is a plug-and-play [ComfyUI](https://github.com/comfyanonymous/ComfyUI) model patch that improves UNet diffusion by gently reordering attention tokens rather than changing weights. It mixes keys and values through learnable, locality-aware transports so structure stays stable while textures sharpen and clutter reduces. The node uses four ingredients: a log-sigma-aware schedule that decides when to act; a multi-scale optimal-transport blend (scales 2, 4, 8) that preserves both edges and layout; a KL-budgeted mixer that automatically limits how far keys can drift; and an exponential-moving-average momentum that prevents flicker across steps. Late in sampling it fades key mixing to zero and keeps only a light value mix, preserving detail without breaking composition. Stabilizes frame-to-frame behavior at the same prompt and seed. 

# Mathematics

Inputs and shapes

q ∈ R^(B×T×Dq), k ∈ R^(B×T×Dk), v ∈ R^(B×T×Dv). If Dq ≠ Dk, q or k is projected once to match dimension using a cached orthonormal basis. Baseline attention: A0 = softmax((q kᵀ) / sqrt(D)), computed per head and batch.

Schedule and intensity

Let σ be the current noise level. Map to u in [0,1] by normalizing log σ between smin and smax. Define a bell-shaped step weight w(u) = 4 u (1 − u). Estimate uncertainty from A0 by entropy; compress to a scalar h in [0,1]. The raw mix amplitude a = intensity × h × w(u).

Multi-scale transport

Tokens lie on a H×W grid with T = H·W. For each scale s in {2,4,8}, build a locality radius r = s and a masked cost on the grid using L1 distance. Convert cost to a kernel and apply a few Sinkhorn normalizations to obtain a near-doubly-stochastic transport P_s ∈ R^(T×T). Blend transports with data-driven weights that depend on u: w₂ = 0.5(1−u), w₄ = u(1−u), w₈ = u, then renormalize so w₂ + w₄ + w₈ = 1. Form P = w₂ P₂ + w₄ P₄ + w₈ P₈ for keys and values separately (Pk, Pv), followed by a short rebalance so rows and columns remain near 1.

Momentum across steps

Maintain EMA transports to avoid temporal jitter: Pk ← β Pk_prev + (1−β) Pk, Pv ← β Pv_prev + (1−β) Pv, with β ≈ 0.8, then rebalance again.

KL-budgeted key mixing

Compute a candidate key permutation k_perm = Pk k. Choose the smallest α_k ∈ [0, α_max] by binary search such that the divergence between new and old attentions stays under a cap that shrinks as denoising progresses: KL(A(q, (1−α_k)k + α_k k_perm) || A0) ≤ κ(u). This prevents over-shuffle while still allowing visible improvements.

Late-step value anneal

Form v_perm = Pv v. Use an amplitude a_v that fades with u so that key mixing stops near the end (u ≥ u_stop) while value mixing remains softly active to enhance texture: a_v = a_base × g(u), with g(u) decreasing from about 1 to about 0.5.

Final updates

k ← (1 − α_k) k + α_k k_perm
v ← (1 − a_v) v + a_v v_perm
q is rescaled by a mild temperature derived from u to keep logits well-conditioned. Attention then proceeds with A = softmax((q kᵀ) / sqrt(D)) and output = A v.

Why it works

Multi-scale transport preserves both local edges (small scales) and global layout (large scales). The KL budget keeps the new attention close to the original where the model is confident, avoiding artifacts. EMA maintains coherence across steps. Late-step value anneal adds texture without disturbing the final composition, yielding sharper, cleaner images with stable structure.

## Install

1. **Clone the node**

	Go to the custom_nodes directory.
	Run this command.
	```bash
	git clone https://github.com/geltz/sws
	```
3. **Restart [ComfyUI](https://github.com/comfyanonymous/ComfyUI).**

5. **Use the node**

   Add **“Sigma-Weighted Shuffle”** (category: `model_patches`) before your sampler; connect the `MODEL` input and set `intensity` (0–1).

   Recommended strength: `intensity = 0.5`.




