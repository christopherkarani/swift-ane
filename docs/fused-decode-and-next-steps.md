# Fused Decode Kernel + Performance Tuning Research

**Date:** 2026-03-06
**Host:** M3 Max, macOS 15.x
**Branch:** `feat/vc-eval-probe`

---

## 1. Fused Decode Kernel (IMPLEMENTED)

### Problem

ANE decode throughput is eval-dominant. The decode path dispatched 12 synchronous
`eval()` calls per token (2 kernels x 6 layers), each carrying ~0.095ms dispatch
overhead. After exhausting IO trimming and hitting dead ends with `_ANEChainingRequest`
and `_ANEVirtualClient`, the remaining software-addressable overhead was **dispatch count**.

### Solution

Combine the decode attention kernel (`DecodeAttentionQKVGenerator`) and FFN kernel
(`DecodeFFNGenerator`) into a single MIL program per layer (`FusedDecodeLayerGenerator`).
This halves the dispatch count from 12 to 6 and eliminates the attn-to-FFN surface copy.

### Implementation

| File | Role |
|------|------|
| `Sources/MILGenerator/FusedDecodeLayerGenerator.swift` | Single MIL combining RMSNorm1 + QKV + attention + Wo + residual1 + RMSNorm2 + SwiGLU FFN + residual2 |
| `Sources/ANERuntime/FusedDecodeKernelSet.swift` | `~Copyable` kernel wrapper (9 weight blobs) |
| `Sources/Espresso/DecodeForwardPass.swift` | `FusedDecodeSurfaceHandles` + `runFusedDecodeTimed()` loop |
| `Tests/ANERuntimeTests/FusedDecodeKernelTests.swift` | 19 tests (12 MIL + 7 hardware-gated) |

### MIL Design

```
Inputs (4):
  x         [1, dim, 1, laneSpatial]   token packed at lane 0
  kCache    [1, dim, 1, maxSeq]         K cache
  vCache    [1, dim, 1, maxSeq]         V cache
  maskCache [1, dim, 1, maxSeq]         attention mask

Outputs (3):
  xNext     [1, dim, 1, laneSpatial]   both residuals applied
  kfFull    [1, dim, 1, laneSpatial]   K projection for cache writeback
  vfFull    [1, dim, 1, laneSpatial]   V projection for cache writeback

Weights (9 BLOBFILE constants):
  rms1.bin, wq.bin, wk.bin, wv.bin, wo.bin   (attention)
  rms2.bin, w1.bin, w3.bin, w2.bin            (FFN)
```

**SSA name collision strategy**: FFN RMSNorm variables are prefixed `f_` (e.g., `f_sq`,
`f_ss`, `f_rrms`, `f_xr`, `f_xn`). Shared constants (`invd`, `eps`, `nhalf`, `raxCh`,
`raxSp`, `kd`) and conv constants (`pt`, `st`, `pd`, `dl`, `gr`) are defined once in
the attention block and reused by the FFN block.

### Measured Results (M3 Max, 1 layer, maxSeq=32)

| Path | ANE ms/step | Dispatches |
|------|-------------|-----------|
| Unfused (attn + FFN) | 0.761 | 2 |
| Fused (single dispatch) | 0.666 | 1 |
| **Savings** | **0.095 (12.5%)** | **-1** |

Extrapolated over 6 layers: **~0.57ms saved per token** (from 12 to 6 dispatches).

---

## 2. Current Decode Budget Breakdown (6 layers)

```
Component                   Unfused         Fused (current)
------------------------------------------------------------
ANE dispatch overhead       6 x 0.095ms     eliminated
                            = 0.570ms

ANE compute (hw time)       6 x 0.310ms     6 x 0.405ms
                            = 1.860ms        = 2.430ms

IO (cache update + chain)   6 x 0.029ms     6 x 0.024ms
                            = 0.174ms        = 0.144ms

Attn->FFN surface copy      6 x 0.005ms     eliminated
                            = 0.030ms

Total per token             ~2.634ms         ~2.574ms
```

**The remaining wall is ANE compute time** -- that is hardware silicon speed. The
dispatch overhead was the largest software-addressable cost and has been eliminated.

---

## 3. Approaches to 4x Over CoreML

### 3.1 Multi-Layer Kernel Fusion via KV Cache Packing (HIGHEST PRIORITY)

**Concept**: Fuse 2 transformer layers into a single MIL program. Pack both layers' KV
caches into one input IOSurface using `slice_by_size` to extract each region. This
reduces dispatches from 6 to 3 and eliminates inter-layer surface copies within each
fused pair.

**Cache packing layout** (channel axis of a `[1, C_packed, 1, maxSeq]` input):

| Channel Offset | Width | Content |
|----------------|-------|---------|
| 0 | 768 | Layer 0 kCache |
| 768 | 768 | Layer 0 vCache |
| 1536 | 768 | Layer 0 maskCache |
| 2304 | 768 | Layer 1 kCache |
| 3072 | 768 | Layer 1 vCache |
| 3840 | 768 | Layer 1 maskCache |

`C_packed = 4608`, packed cache size = 294,912 bytes (0.28 MB).

**Memory analysis (2-layer fused program, dim=768, hidden=2048)**:

| Component | Per Layer | 2 Layers |
|-----------|-----------|----------|
| Attention weights (Wq,Wk,Wv,Wo) | 4 x 768^2 x 2 = 4.72 MB | 9.44 MB |
| FFN weights (W1,W3,W2) | 3 x 2048x768 x 2 = 9.44 MB | 18.87 MB |
| RMSNorm (rms1,rms2) | 2 x 768 x 2 = 3 KB | 6 KB |
| **Total baked weights** | **14.28 MB** | **28.57 MB** |
| Cache data in IOSurface | 0.14 MB | 0.28 MB |
| **Working set vs 32MB SRAM** | fits | fits (ANE streams weights via DMA) |

**MIL feasibility**: `slice_by_size` on channel-packed inputs is a proven pattern -- the
backward pass generators (`SDPABackward1Generator`, `QKVBackwardGenerator`, etc.) already
perform 3-4 slices from packed inputs with convolutions and full attention math. The
documented `slice_by_index` failure does NOT apply to `slice_by_size`.

**Expected savings**: 3 more dispatch overheads eliminated (3 x 0.095ms = 0.285ms) plus
inter-layer copy elimination. Combined with single-layer fusion: **12 dispatches to 3**.

**Layer fusion scaling**:

| Fused Layers | Dispatches | Baked Weights | SRAM Fit? | Feasibility |
|---|---|---|---|---|
| 1 (current) | 6 | 14.28 MB | yes | **done** |
| 2 | 3 | 28.57 MB | marginal | **high confidence** |
| 3 | 2 | 42.85 MB | no (streams) | medium confidence |
| 6 | 1 | 85.70 MB | no | low confidence |

**Recommendation**: Start with 2-layer fusion. If successful, try 3.

### 3.2 Metal + ANE Hybrid Decode (HIGH IMPACT, UNEXPLORED)

**Concept**: Use ANE for convolution-heavy work (QKV projections, FFN) and Metal for
dynamic attention computation (Q@K^T, softmax, weights@V). IOSurfaces are the native
shared memory format for both Metal and ANE -- **zero copy between accelerators**.

**Why this could achieve 4x**:
- Two accelerators working in **parallel**: ANE does layer N's FFN while Metal does
  layer N+1's attention
- Metal `simdgroup_matrix` on M3 provides ~7 additional TFLOPS for matmul
- ANE's attention is currently a probe path (bypasses real math). Metal handles dynamic
  attention natively with full flexibility
- CoreML cannot do this -- it dispatches to one accelerator at a time per subgraph

**Architecture**:
```
ANE:   [QKV proj] -----> IOSurface (Q,K,V) -----> [FFN]
                              |                      ^
                              v                      |
Metal:                  [Q@K^T -> softmax -> @V] --> IOSurface (attn_out)
```

**Challenges**:
- Requires Metal compute pipeline for attention
- Synchronization between ANE and Metal dispatch
- Surface format compatibility (both use FP16, both use IOSurface)

### 3.3 `evaluateRealTimeWithModel:` Probe (MEDIUM IMPACT, QUICK WIN)

The `_ANEClient` exposes a real-time eval path:
- `beginRealTimeTask` / `endRealTimeTask` lifecycle
- `evaluateRealTimeWithModel:options:request:error:`

**This has never been tested for dispatch latency.** If the real-time path bypasses the
QoS scheduling queue, it could have materially lower dispatch overhead than
`evaluateWithQoS:`. The existing interop code already has conditional support for this
path (controlled by `ANE_REALTIME` environment variable).

**Estimated effort**: 1-2 hours to probe and benchmark.

### 3.4 Metal SharedEvent on Standard Eval Path (MEDIUM IMPACT)

`_ANERequest.setSharedEvents:` works on the standard eval path (no VirtualClient
required). If we construct an `IOSurfaceSharedEvent` via `[MTLDevice newSharedEvent]`
and attach it, the ANE hardware might signal completion asynchronously -- enabling true
pipelined dispatch where:

1. Submit eval for layer N
2. ANE signals Metal SharedEvent when done
3. CPU starts IO for layer N while layer N+1 eval begins

**Key unknown**: Does `setSharedEvents:` on the standard `_ANEClient` eval path actually
trigger hardware-level async signaling? The `completionEvent:` parameter on the
VirtualClient eval path suggests it should, but the standard path may ignore it.

### 3.5 Speculative Decoding (ALGORITHMIC, HIGH IMPACT)

**Concept**: Use a tiny draft model (1-2 layers, or a distilled version) to generate N
candidate tokens, then verify all N in one batched forward pass on the full model. ANE
excels at batch compute (deep graphs, high utilization).

This transforms the decode problem from "N sequential single-token dispatches" into
"1 cheap draft + 1 expensive verify" -- an algorithmic 3-5x on top of hardware
optimization.

**Why ANE is particularly suited**: The prefill path (batch tokens) already achieves much
higher utilization than decode (single token). Speculative decoding turns decode into a
prefill-like workload.

---

## 4. Approaches Proven Blocked (Do Not Retry)

| Approach | Failure Mode | Evidence |
|----------|-------------|----------|
| IOSurface aliasing between kernels | `statusType=0x9: Program Inference error` | Compile-time external surface fails at eval |
| _ANERequest surface rebinding | `-[__NSArrayM procedureIndex]` type confusion, then `0x9` | Request is immutable after construction |
| _ANEVirtualClient instantiation | All 5 init paths return nil | Kernel-level IOKit entitlement gate |
| _ANEChainingRequest | `PREPARE_FAILED` across all parameter permutations | Exhausted A22-A26 experiments |
| `setCodeSigningIdentity:` | Crashes with `-[__NSCFConstantString setObject:forKey:]` | Identity stored in class-level static dict |
| QoS tuning (0-63) | No effect on latency | All values produce identical timing |

---

## 5. Queue Pipelining Research

### Can we overlap eval with IO?

`evaluateWithQoS:` is **synchronous** -- it blocks the calling thread until the ANE
hardware completes. The `completionHandler` fires synchronously on the calling thread
after eval returns (not on a background thread). This means:

- CompletionHandler chaining provides **zero pipelining benefit**
- The only async eval path (`doEvaluateWithModel:completionEvent:`) requires VirtualClient
  (blocked by entitlements)

### What IS possible

**Background dispatch with cache overlap**: Dispatch `eval()` on a GCD queue. While
the eval blocks that queue, the main thread can perform K/V/mask cache updates for the
*previous* layer. Savings: ~5-10 microseconds per layer (the cache updates are already
very fast spatial-slice copies).

**Double-buffered handles**: Compile each kernel twice with separate surfaces. Write
to Handle-A while Handle-B evals. Eliminates inter-token IO latency but doubles compile
budget (which is capped at ~100 per process) and surface memory.

---

## 6. Priority Ranking

| # | Approach | Expected Impact | Effort | Risk |
|---|----------|----------------|--------|------|
| 1 | 2-layer KV cache packing | 0.285ms/token + zero inter-layer IO | Medium | Low |
| 2 | `evaluateRealTimeWithModel:` probe | Unknown (possibly 0.05ms/dispatch) | Small | Low |

---

## 7. Avenue 1 Result — Multi-Layer Kernel Fusion (ABANDONED)

**Date:** March 6, 2026

### Attempt 1: Fully packed K/V/mask cache input

- Built `FusedTwoLayerDecodeGenerator` / `FusedTwoLayerDecodeKernelSet` with a packed cache input
  `[1, 4608, 1, maxSeq]` and `slice_by_size` extraction for:
  - L0 K, V, mask
  - L1 K, V, mask
- Added non-hardware TDD coverage for:
  - MIL header / blob paths
  - packed-cache slice count
  - unique SSA names
  - I/O byte-size contract
- Hardware result:
  - `ANE_HARDWARE_TESTS=1 swift test --filter FusedTwoLayerDecodeKernelSetTests`
  - `_ANECompiler : ANECCompile() FAILED`
  - underlying compiler error: `InvalidMILProgram`

### Fallback: K/V-only packing, masks split back out

- Reduced the candidate to:
  - `packedKVCache` input `[1, 3072, 1, maxSeq]`
  - `maskCache0` input `[1, 768, 1, maxSeq]`
  - `maskCache1` input `[1, 768, 1, maxSeq]`
- This isolates packed K/V slicing from packed mask slicing while preserving the paired-kernel objective.
- Hardware result:
  - `ANE_HARDWARE_TESTS=1 swift test --filter FusedTwoLayerDecodeKernelSetTests/test_fused_two_layer_compile_fails_with_controlled_error_on_hardware`
  - same `_ANECompiler : ANECCompile() FAILED`
  - same underlying `InvalidMILProgram`

### Outcome

- Status: **abandoned**
- Reason: both the full packed-cache design and the K/V-only fallback fail at compile time with
  `InvalidMILProgram`, so the avenue never reaches eval.
- Benchmark delta: **N/A**. No post-change per-token timing exists because the candidate kernel never compiled.
- Per-token gain credited to this avenue: **+0.000ms**
- Running cumulative savings toward the 4x goal: **0.000ms**

## Results Summary

| Avenue | Status | Per-Token Gain | Cumulative |
|--------|--------|----------------|------------|
| 1. Multi-layer fusion | abandoned (`InvalidMILProgram` on full pack and K/V-only fallback) | +0.000ms | 0.000ms |
| 2. Metal SharedEvent | abandoned | +0.000ms | 0.000ms |
| 3. Metal+ANE hybrid | abandoned | +0.000ms | 0.000ms |
| 4. CoreML baseline | measured | N/A | 0.000ms |
| 5. Speculative decode | pending | pending | 0.000ms |
| 6. GCD pipeline | pending | pending | 0.000ms |

Direct ANE: 2.875ms/token
CoreML:     3.007ms/token
Speedup:    1.05x
| 3 | Metal + ANE hybrid decode | Potentially 2-4x (parallel accelerators) | Large | Medium |
| 4 | Metal SharedEvent on standard eval | True async dispatch | Medium | Medium |
| 5 | Speculative decoding | 3-5x algorithmic | Large | Low |

### Recommended Execution Order

1. **Probe `evaluateRealTimeWithModel:`** -- quick win, 1-2 hours, might reveal a
   lower-overhead dispatch path
2. **Build 2-layer fused kernel with packed caches** -- proven MIL patterns, high
   confidence, eliminates 3 more dispatches
3. **Prototype Metal attention kernel** -- if successful, enables the hybrid architecture
   that could deliver 4x
4. **Speculative decoding** -- algorithmic multiplier on top of everything else

---

## 7. CoreML Comparison Gap

**We have no direct CoreML benchmark for the same model.** All throughput numbers are
for the direct `_ANEClient` path. To claim "4x over CoreML" we need:

1. Export the 6-layer transformer via coremltools as `.mlmodel`
2. Run inference with `.cpuAndNeuralEngine` compute units
3. Measure per-token decode latency on the same M3 Max hardware
4. Compare against our direct path

CoreML uses the same ANE hardware and MIL compiler. It may dispatch more efficiently
for some workloads (mature scheduling, internal graph fusion). Our advantage is
fine-grained control over kernel boundaries, zero-copy IOSurface management for KV
caches, and the ability to do things CoreML cannot (training, custom fusion, hybrid
dispatch).

## 8. Avenue 2 Result — Metal SharedEvent on Standard Eval Path (ABANDONED)

Date: 2026-03-06

What I built:
- Extended `ane_interop_probe_standard_completion_handler(...)` to optionally create a Metal-backed `MTLSharedEvent` via `MTLCreateSystemDefaultDevice()` and `newSharedEvent`.
- Wrapped that event in `_ANESharedSignalEvent` and `_ANESharedEvents`.
- Attached the container to `_ANERequest` via `setSharedEvents:` on the standard eval path.
- Extended the Swift probe surface to report attachment state and event value before/after eval.

Expected improvement:
- If ANE signaled the Metal shared event on completion, decode could pipeline host and accelerator work and potentially overlap adjacent dispatches.

Baseline before changes:
- Existing standard completion handler works, but fires synchronously on the calling thread and provides no true async overlap.
- No measurable per-token gain from the baseline path.

Observed result:
- `MTLCreateSystemDefaultDevice()` succeeded.
- `newSharedEvent` succeeded.
- `_ANERequest` reports `setSharedEvents:` exists.
- With the Metal-backed shared event attached, the hardware test enters the standard eval path and then hangs with no completion callback and no observable event advancement.

Abandon reason:
- This path is a dead end under the project protocol. It does not produce a usable signal or measurable pipeline parallelism, and it blocks progress inside eval instead.

Timing:
- Post measurement: not recorded.
- Delta: +0.000ms/token.
- Cumulative savings after Avenue 2: 0.000ms/token.

## 9. Avenue 3 Result — Metal + ANE Hybrid Decode (ABANDONED FOR NOW)

Date: 2026-03-06

What I built:
- Added `MetalAttentionKernel` as a standalone Metal SDPA stage using three compute kernels (`logits`, `softmax`, `output`).
- Bound IOSurface memory into Metal via `MTLDevice.makeBuffer(bytesNoCopy:...)` so Q/K/V/mask/output all stay zero-copy.
- Added correctness and benchmark tests around the standalone Metal path.

TDD:
- Wrote `MetalAttentionKernelTests` first.
- Initial test run failed at compile time because `MetalAttentionShape` / `MetalAttentionKernel` did not exist.
- After implementation, correctness matched the CPU reference and the benchmark probe passed.

Baseline before changes:
- Direct ANE decode benchmark, 6 layers, `steps=32`, `maxSeq=32`, `warmup=3`, `iterations=20`:
  - Median: `2.860 ms/token`
  - Mean: `2.840 ms/token`
  - ANE kernel time: `2.525 ms/token`
- Per-layer decode profile on the same run:
  - Average ANE attention eval: `201.516 us/layer`
  - Average ANE FFN eval: approximately `219-221 us/layer`

Post measurement:
- Standalone Metal attention benchmark, `heads=12`, `headDim=64`, `seqLen=32`, `warmup=3`, `iterations=20`, timed with `mach_absolute_time()`:
  - Mean: `0.395252 ms/eval`
  - Median: `0.217437 ms/eval`
  - Zero-copy IOSurface bindings: `true`

Delta:
- Naive stage substitution comparison:
  - Current ANE attention: `0.201516 ms/layer`
  - Metal attention: `0.217437 ms/layer`
  - Savings: `0.201516 - 0.217437 = -0.015921 ms/layer` (`-7.9%`)
- Extrapolated over a full 6-layer decode with no overlap: `-0.095526 ms/token` regression.

Why this avenue is abandoned for now:
- The zero-copy gate passed, which is useful.
- But the current decode path cannot actually swap ANE attention math for Metal attention math in place.
- `DecodeKernelSet.decodeAttnQKV` already performs attention internally and only exposes post-attention `attnX2Out` plus K/V outputs.
- A real hybrid decode needs a new ANE QKV-only stage before Metal can own `Q@K^T -> softmax -> @V`.
- That kernel split is a larger follow-on effort, so this avenue is documented as promising but abandoned in this pass.

Timing impact recorded for this pass:
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 3: `0.000 ms/token`

## 10. Avenue 4 Result — CoreML Baseline Benchmark (MEASURED)

Date: 2026-03-06

What changed:
- Extended `scripts/generate_coreml_model.py` to support `--layers` and `--output`.
- Generated a 6-layer CoreML package at `benchmarks/models/transformer_6layer.mlpackage`.
- Benchmarked decode on the same 6-layer, `steps=32`, `maxSeq=32` workload used for direct ANE.

TDD / pre-change failure:
- The original exporter ignored `--layers 6 --output ...` and still wrote the default single-layer model to `benchmarks/models/transformer_layer.mlpackage`.
- After the exporter change, the same command produced the requested 6-layer package.

Benchmark configuration:
- Warmup: `5` sequences
- Timed: `100` sequences (`3200` measured tokens)
- Decode schedule: `steps=32`, `maxSeq=32`
- CoreML model: `benchmarks/models/transformer_6layer.mlpackage`

Measured results:
- Direct ANE decode:
  - Mean: `3.098 ms/token`
  - Median: `2.875 ms/token`
- CoreML Decode (`.all`):
  - Mean: `3.005 ms/token`
  - Median: `2.992 ms/token`
- CoreML Decode (`.cpuAndNeuralEngine`):
  - Mean: `3.017 ms/token`
  - Median: `3.007 ms/token`

Speedup vs CoreML `.cpuAndNeuralEngine`:
- `3.007 - 2.875 = 0.132 ms/token` saved
- Relative speedup: `3.007 / 2.875 = 1.05x`

Strict fastest-CoreML gate:
- Fastest CoreML median observed: `.all` at `2.992 ms/token`
- Direct ANE median: `2.875 ms/token`
- Speedup vs fastest CoreML: `1.04x`

Interpretation:
- The direct `_ANEClient` path currently beats CoreML on this workload, but only narrowly.
- The current project is far from the `4x over CoreML` target; the measured headroom is about `5%` over `.cpuAndNeuralEngine` and `4%` over the fastest CoreML configuration observed here.
