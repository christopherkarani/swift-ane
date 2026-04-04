# autoresearch-espresso

Autonomous experimentation on ANE kernel optimizations for **stories110m** decode throughput (tok/s) with quality.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr4`). The branch `autoresearch-espresso/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-espresso/<tag>` from current main.
3. **Read the in-scope files**: The Espresso repo is large. Focus on these:
   - `Sources/ANEBuilder/` — kernel graph builders (attention, FFN, composites, primitives)
   - `Sources/ANEPasses/` — optimization passes (identity elimination, cast elimination, dead code)
   - `Sources/ANERuntime/` — ANE compilation, kernel sets, decode paths
   - `Sources/Espresso/DecodeForwardPass.swift` — THE decode loop, env var knobs
   - `Sources/MILGenerator/` — MIL text generators for ANE kernels
   - `Sources/EspressoBench/` — benchmark CLI
   - `experiment_runner.py` — the experiment runner (build, benchmark, quality check)
4. **Verify build works**: `swift build --product espresso-bench` should succeed.
5. **Establish baseline**: Run `python experiment_runner.py benchmark` to get the current tok/s baseline.
6. **Save reference generations**: Run `python experiment_runner.py quality-check` and save the output to `autoresearch-espresso/reference_generations.json`.
7. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment modifies Swift source files, rebuilds, and benchmarks. The training script runs for a **fixed benchmark configuration** — same prompts, same decode steps, same iterations. You launch it via `python experiment_runner.py full`.

**What you CAN do:**
- Modify any Swift source file in `Sources/ANEBuilder/`, `Sources/ANEPasses/`, `Sources/ANERuntime/`, `Sources/Espresso/`, `Sources/MILGenerator/` — anything that affects ANE kernel generation, compilation, or decode performance. Everything is fair game: kernel fusion, memory layout, optimization passes, env var defaults, compile strategies, etc.

**What you CANNOT do:**
- Modify `experiment_runner.py`. It is the fixed evaluation harness.
- Modify `Package.swift` — do not add dependencies or change build structure.
- Modify `ModelConfig.swift` — model dimensions (dim, heads, layers, vocab) are fixed.
- Change the benchmark prompts, decode steps, or iteration counts.
- Install new packages or add dependencies.

**The goal is simple: get the highest tok/s (tokens per second) while maintaining generation quality.** The time budget is implicit — each experiment should take ~5-10 minutes (build + benchmark). Everything is fair game: kernel fusion, memory optimization, pass improvements, compilation strategies. The only constraint is that the code builds without crashing, the benchmark runs, and generation quality is maintained.

**Quality gate**: Generated text on the 4 fixed prompts must match the reference generations exactly (argmax is deterministic). If quality regresses, the experiment is a **discard**.

**Simplicity criterion**: All else being equal, simpler is better. A small tok/s improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

## Key Experiment Knobs

These are the primary areas to experiment with:

### 1. `ESPRESSO_DECODE_LANE_SPATIAL` (HIGH IMPACT)
- **File**: `Sources/ANERuntime/DecodeKernelSet.swift` (and all kernel sets with `resolvedLaneSpatialForCurrentProcess()`)
- **Default**: `32`
- **Try**: `16`, `32`, `64`, `128`
- This directly affects tile size, IOSurface sizes, and kernel execution time. The single most impactful knob.

### 2. Kernel Set Selection (HIGH IMPACT)
- **Files**: `Sources/Espresso/DecodeForwardPass.swift`, various `*KernelSet.swift` files
- **Options**:
  - `DecodeKernelSet` — separate attention + FFN kernels per layer
  - `FusedDecodeKernelSet` — single fused kernel per layer (eliminates inter-kernel round-trip)
  - `FusedTwoLayerDecodeKernelSet` — two layers in one kernel
  - `HybridDecodeKernelSet` — ANE QKV + Metal attention + ANE FFN
- Try changing the default kernel set, or adding new fusion strategies.

### 3. Metal Fused SDPA (HIGH IMPACT)
- **Env var**: `ESPRESSO_DISABLE_METAL_FUSED_SDPA`
- **File**: `Sources/Espresso/DecodeForwardPass.swift`
- Toggle between Metal fused SDPA vs. pure ANE attention. The Metal path may be faster for some configurations.

### 4. Hybrid Fused Post-Attention (MEDIUM IMPACT)
- **Env var**: `ESPRESSO_DISABLE_HYBRID_FUSED_POST_ATTENTION`
- **File**: `Sources/ANERuntime/HybridDecodeKernelSet.swift`
- Toggle fused vs. split projection+FFN in the hybrid path.

### 5. ANE Optimization Passes (MEDIUM IMPACT)
- **Directory**: `Sources/ANEPasses/`
- **Ideas**:
  - Add new fusion passes (e.g., fuse consecutive element-wise ops)
  - Improve CastElimination to handle more patterns
  - Add constant folding for static graph portions
  - Add memory reuse passes (reuse buffers across layers)
  - Reorder pass pipeline for better fixpoint convergence
  - Increase `ANEOptimizationPipeline.maxIterations` beyond 20

### 6. MIL Generation (MEDIUM IMPACT)
- **Directory**: `Sources/MILGenerator/`
- **Directory**: `Sources/ANECodegen/`
- **Ideas**:
  - Change how constants are emitted (inline vs. const() nodes)
  - Experiment with `deploymentTarget` (`"ios18"` → `"macos15"` etc.)
  - Optimize MIL patterns for frequently-used ops
  - Change MIL header constants for different compiler behavior

### 7. Cache/Surface Management (MEDIUM IMPACT)
- **File**: `Sources/Espresso/DecodeForwardPass.swift`
- **Env vars**: `ESPRESSO_DECODE_FORCE_FULL_WINDOW_SYNC`, `ESPRESSO_REPACK_VOUT_HEAD_MAJOR`
- **Ideas**: Reduce KV cache synchronization overhead, optimize surface layout

### 8. Compile Strategies (LOWER IMPACT)
- **Env var**: `ESPRESSO_DISABLE_HYBRID_DONOR_DELTA`
- **File**: `Sources/ANERuntime/ANEKernel.swift`
- **Ideas**: Delta compilation reuse, compile budget management, retry policies

### 9. ANEBuilder Kernel Construction (LOWER IMPACT)
- **Directory**: `Sources/ANEBuilder/`
- **Ideas**:
  - Fuse operations in graph construction
  - Change how RMSNorm is built (FP16 vs FP32 path)
  - Try different activation function implementations
  - Optimize composite kernel patterns

## Output format

Once the benchmark finishes, it outputs tok/s metrics. You can extract the key metric from the experiment runner output:

```
[bench] tok/s: XXX.X, ms/tok: XX.X
[quality] X/4 matched
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	tok_s	ms_per_tok	compile_time_ms	quality	status	description
```

1. git commit hash (short, 7 chars)
2. tok/s achieved (e.g. 519.3) — use 0.0 for crashes
3. ms per token (e.g. 1.93) — use 0.0 for crashes
4. compile time in ms (e.g. 12000) — use 0 for crashes
5. quality score (0.00-1.00, exact match ratio on 4 prompts)
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	tok_s	ms_per_tok	compile_time_ms	quality	status	description
a1b2c3d	153.0	6.54	45000	1.00	keep	baseline: direct transformer decode
b2c3d4e	519.0	1.93	52000	1.00	keep	switch to fused decode kernel set
c3d4e5f	480.0	2.08	50000	1.00	discard	lane spatial 64 (slower)
d4e5f6g	0.0	0.0	0	0.0	crash	aggressive fusion (build failed)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-espresso/apr4`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune Swift source files with an experimental idea by directly hacking the code.
3. `git commit -m "experiment: <description>"`
4. Run the experiment: `python experiment_runner.py full > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "\[bench\] tok/s\|\[quality\]" run.log`
6. If the grep output is empty or shows build failure, the run crashed. Run `tail -n 50 run.log` to read the error and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If tok/s improved (higher) AND quality == 1.00, you "advance" the branch, keeping the git commit
9. If tok/s is equal/worse or quality dropped, git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5-10 minutes total (build + benchmark). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (build failure, benchmark hang, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up in the morning to experimental results, all completed by you while they slept!

## Experiment Idea Starter Pack

Here are some concrete ideas to get started:

1. **Lane spatial sweep**: Try `ESPRESSO_DECODE_LANE_SPATIAL` = 16, 32, 64, 128 in each kernel set
2. **Fusion experiments**: Fuse attention + FFN into single kernel (eliminate intermediate surface round-trip)
3. **Pass improvements**: Add a pass that fuses consecutive element-wise ops (add+mul → single op)
4. **MIL constant inlining**: Emit small constants inline instead of as const() nodes — may speed up compilation
5. **Cast elimination**: Expand CastEliminationPass to handle more patterns (e.g., fp32→fp16→fp32 chains)
6. **Memory reuse**: Add a pass that reuses surface buffers across layers when shapes match
7. **Dead code elimination**: Run DCE more aggressively — eliminate unused graph branches earlier
8. **SRAM budget optimization**: Restructure graphs to stay under 32MB SRAM (avoid DRAM spill ~30% penalty)
9. **Kernel set hybridization**: Mix different kernel types for different layers (e.g., fused for early layers, separate for late)
10. **Metal vs ANE attention**: Toggle `ESPRESSO_DISABLE_METAL_FUSED_SDPA` and benchmark both paths
