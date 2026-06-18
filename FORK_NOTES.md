# FL AceStep Training PLUS — memory-management fork

**`ComfyUI-FL-AceStep-Training-PLUS`** is a fork of
[filliptm/ComfyUI-FL-AceStep-Training](https://github.com/filliptm/ComfyUI-FL-AceStep-Training)
that surfaces VRAM / CPU-RAM controls which the upstream node hardcodes, so users
on lower-spec cards can tune them. The node `class_type` keys are unchanged, so
it is a **drop-in replacement** — existing workflows load unchanged, and the two
enhanced nodes show as **"(PLUS)"** in the menu. Three files differ from upstream
(`nodes/training_config.py`, `nodes/training_ui.py`, `requirements.txt`).

## What changed

### `nodes/training_config.py`
Three new inputs on the **Training Configuration** node (in the `optional`
group, so existing workflows keep working unchanged):

| Input | Type | Default | What it does |
|---|---|---|---|
| `optimizer_type` | enum | `AdamW (fused)` | Choose the optimizer (see below). |
| `gradient_checkpointing` | bool | `False` | Recompute activations in the backward pass — biggest single VRAM saver, ~20-30% slower. |
| `empty_cache_every_n_steps` | int | `0` (off) | Periodically `torch.cuda.empty_cache()` to fight VRAM fragmentation on long runs. |

`optimizer_type` options:
- **AdamW (fused)** — the original upstream default (fused kernel on CUDA, fastest, most VRAM).
- **AdamW (standard)** — torch AdamW without the fused kernel (slightly less VRAM).
- **AdamW8bit (bitsandbytes)** — 8-bit optimizer state, ~2× smaller optimizer memory.
- **PagedAdamW8bit (CPU-paged, bitsandbytes)** — 8-bit **and pages optimizer state to
  CPU RAM**. This is the "offload to system RAM during training" option — the
  lowest-VRAM choice for tight cards.

### `nodes/training_ui.py`
- `_enable_gradient_checkpointing()` after PEFT injection: calls
  `enable_input_require_grads()`, tries the native `gradient_checkpointing_enable()`
  hook, and **if there is none, falls back to structure-agnostic manual wrapping**
  (`_wrap_transformer_blocks()`) — it discovers the DiT's transformer-block
  `ModuleList`(s) (homogeneous stacks whose type name contains "block"/"layer",
  skipping nested ones) and wraps each block's `forward` in
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. The wrapper
  passes through when grad is disabled, so inference/validation is unaffected.
  The console logs how many blocks were wrapped (or warns if none were found).
- Replaces the hardcoded fused-AdamW setup with `_build_optimizer()`, which honors
  `optimizer_type` and **falls back to AdamW with a clear warning if bitsandbytes
  is missing or has no kernel for your GPU**.
- Calls `_maybe_empty_cache()` after each optimizer step.

### `requirements.txt`
Adds a commented-out optional `bitsandbytes` line (the 8-bit optimizers need it;
the node degrades gracefully without it).

## Install

Use this fork **instead of** the upstream pack (same node names — don't run both):

```bash
cd ComfyUI/custom_nodes
# remove/disable the upstream ComfyUI-FL-AceStep-Training first if present
pip install -r ComfyUI-FL-AceStep-Training-PLUS/requirements.txt
# for the 8-bit / CPU-paged optimizers:
pip install bitsandbytes        # use a build matching your GPU + CUDA
```

## Recommended low-VRAM preset

`gradient_checkpointing = True`, `optimizer_type = PagedAdamW8bit`,
`batch_size = 1`, `gradient_accumulation = 8`, `empty_cache_every_n_steps = 50`,
and launch ComfyUI with `--lowvram --reserve-vram 1.0`.

## Honest caveats

- **Not GPU-tested in this fork.** The edits are static-verified (syntax + the
  config node exercised end-to-end) but the training-loop paths need validation on
  a real card with the ACE-Step model + `peft` (and `bitsandbytes` for the 8-bit
  paths). Start with a tiny dataset / few steps to confirm.
- **Gradient checkpointing now auto-wraps** the transformer blocks when there is no
  native hook (the deeper rewrite). Two things to verify on first run: (1) the
  console reports a non-zero "wrapped N blocks" — if it found nothing, the DiT's
  block stack didn't match the "block"/"layer" type-name heuristic and you'll need
  to point it at the right `ModuleList` (tell me the decoder's module names from the
  log and I'll target it precisely); (2) loss still decreases normally — checkpoint
  + a custom flow-matching loop should be fine with `use_reentrant=False`, but
  confirm on a short run.
- **Full base-model CPU offload (layer streaming / block-swap) is *not* included.**
  Doing it correctly for *training* (forward **and** recomputed-backward) needs
  accelerate/DeepSpeed-style offload or a kijai-style block-swap with device
  hooks — a substantial, hard-to-validate-blind rewrite. What this fork gives you
  instead, which covers most of the need on a 16 GB-class card: gradient
  checkpointing (activations) + `PagedAdamW8bit` (optimizer state → CPU RAM) +
  `--lowvram` at the ComfyUI process level (base weights ↔ CPU RAM). If you still
  OOM after those, in-node base-weight block-swap is the next project to scope.

## Upstream

Based on a shallow clone of the upstream `main`. To publish your own fork, re-point
the git remote (`git remote set-url origin <your-fork-url>`) and open a PR upstream
if you'd like these merged back.
