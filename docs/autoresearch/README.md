# Autoresearch

Tracked setup for real-model throughput work on Espresso vs Core ML.

This is intentionally anchored to the shipped GPT-2 compare path on `main`, not the older local-artifact decode benchmark still described in the top-level README.

## Source Of Truth

Use `./scripts/run_autoresearch_compare.sh` as the official benchmark entrypoint.

It wraps `./espresso bench` and exports:

- `compare.json`
- `summary.csv`
- `summary.md`
- `espresso_token_latencies.csv`
- `coreml_token_latencies.csv`

Primary scoreboard:

- `speedup_vs_coreml = espresso.tokens_per_second / coreml.tokens_per_second`

Primary Espresso optimization metric:

- `espresso.tokens_per_second`

Hard gates:

- `token_match == true`
- `text_match == true`

Guardrails:

- `espresso.first_token_latency_ms`
- `espresso.compile_time_ms`
- `espresso.median_token_ms`
- `espresso.p95_token_ms`
- no benchmark contract drift

## Current Throughput Focus

The current real-model GPT-2 path is dominated by three concrete costs in `Sources/RealModelInference/RealModelInferenceEngine.swift`:

- full prompt embedding rebuild plus full input-surface rewrite every token
- full-sequence attention and FFN over the active bucket every token
- full hidden-state readback plus CPU logits every token

Those are the surfaces the implementation lanes target.

The compare bootstrap path still depends on GPT-2 helper scripts referenced by `GPT2DemoSupport.swift`. If your local checkout does not have those helpers available, pin explicit `weights`, `tokenizer`, and `coreml-model` paths in `.autoresearch/env.sh` so the benchmark contract remains runnable.

## Operating Model

- Run exactly one hardware benchmark referee at a time.
- Run implementation agents in separate worktrees.
- Implementation agents do not claim official throughput wins.
- Only referee measurements count.

Recommended lanes:

- `benchmark-referee`
- `incremental-decode`
- `head-path`
- `bucket-io`

## Why This Model

Parallel ANE benchmarks collide on the same hardware and make tokens/sec numbers noisy. The clean pattern is:

1. implementation agents make candidate changes in isolation
2. the benchmark referee validates those changes serially on hardware
3. only referee-validated wins get merged

## Experiment Then Revert

Use the disposable-worktree loop when you want Auto-Inference-Optimiser-style
keep-or-revert behavior without destructive git cleanup:

```bash
# Baseline once
./scripts/run_autoresearch_suite.sh --output-dir /tmp/espresso-baseline

# Candidate in a disposable detached worktree
./scripts/run_autoresearch_experiment.sh \
  --ref feat/my-candidate \
  --baseline /tmp/espresso-baseline/suite-summary.json \
  --env-file ./.autoresearch/env.sh
```

The candidate worktree is deleted automatically when it fails the keep gates.
Passing candidates are kept in place for manual promotion or merge.

## Stories Inner Loop

For the exact Stories110M path, use the fast warm Espresso-only referee before
running a full Core ML compare:

```bash
./scripts/run_stories_generate_benchmark.sh --prompt "Hello"
```

This targets the local Stories artifacts at
`~/Library/Application Support/Espresso/demo/stories110m` by default and keeps
the GPT-2 autoresearch defaults untouched.

LLaMA-family cached bindings remain opt-in until they clear the same keep gate:

```bash
ESPRESSO_ENABLE_LLAMA_HYBRID_CACHED_BINDINGS=1 \
  ./scripts/run_stories_generate_benchmark.sh --prompt "Hello"
```

For exact-head experiments, force the backend explicitly so you only change one
variable per run:

```bash
ESPRESSO_FORCE_EXACT_HEAD_BACKEND=cpu_fp16_tiled \
  ./scripts/run_stories_generate_benchmark.sh --prompt "Hello"
```

Stories now disables hybrid donor delta by default because the delta-reload lane
was responsible for all measured compile retries on this model. Force the old
behavior only when you want to compare it explicitly:

```bash
ESPRESSO_ENABLE_HYBRID_DONOR_DELTA=1 \
  ./scripts/run_stories_generate_benchmark.sh --prompt "Hello"
```

For approximate long-context throughput experiments, cap hybrid decode attention
to a suffix window while keeping the full KV cache intact:

```bash
ESPRESSO_HYBRID_ATTENTION_WINDOW=32 \
  ./scripts/run_stories_generate_benchmark.sh --prompt "long prompt here" --max-tokens 32
```

This mode remains opt-in. It preserves exact output only while the live context
fits inside the configured window.

## Scaffold A Lane

From the repo root or any Espresso worktree:

```bash
./scripts/setup_autoresearch_lane.sh benchmark-referee
./scripts/setup_autoresearch_lane.sh incremental-decode
./scripts/setup_autoresearch_lane.sh head-path
./scripts/setup_autoresearch_lane.sh bucket-io
```

To create and initialize a new worktree in one step:

```bash
./scripts/setup_autoresearch_lane.sh incremental-decode \
  --create-worktree .worktrees/autoresearch-incremental-decode-$(date +%Y%m%d-%H%M%S) \
  --branch feat/autoresearch-incremental-decode
```

The script creates a local, ignored `.autoresearch/` directory in the target worktree with:

- `README.md`
- `program.md`
- `PROMPT.txt`
- `env.sh`
- `bench.sh`

It also creates an ignored `autoresearch-results.tsv` audit log keyed to the real-model throughput metrics. Official referee decisions come from `suite-summary.json` and `judge_suite_results.sh`, not from the raw TSV alone.

## Hardened Contract

The benchmark harness is hardened for trustworthy multi-prompt, multi-run measurement:

- **Prompt suite**: `scripts/benchmark-prompts.txt` defines 3 fixed prompts (short/medium/long) exercising different CoreML sequence-length buckets
- **Suite runner**: `scripts/run_autoresearch_suite.sh` runs 1 cold + K warm runs across all prompts, aggregates median/min/max per prompt, and can forward raw per-invocation rows into `autoresearch-results.tsv`
- **Judge script**: `scripts/judge_suite_results.sh` evaluates single summaries or compares baseline vs candidate with threshold gating

### Merge Bar

A candidate branch is merge-eligible only when:

1. All runs pass `token_match == true` and `text_match == true`
2. Candidate shows >= 2% improvement on `espresso_tok_s` median on **every** prompt
3. The bar holds across all K suite runs (default 3/3)
4. Only `judge_suite_results.sh --baseline` verdicts count

The TSV is an audit trail for each underlying compare invocation. Use it to inspect prompt-level run history, not as the merge decision source.

### Suite Workflow

```bash
# 1. Run baseline suite on main
./scripts/run_autoresearch_suite.sh --runs 3
cp results/autoresearch/suite-*/suite-summary.json /tmp/baseline.json

# 2. Switch to candidate branch, run suite
git checkout feat/my-optimization
./scripts/run_autoresearch_suite.sh --runs 3

# 3. Judge
./scripts/judge_suite_results.sh --baseline /tmp/baseline.json \
  results/autoresearch/suite-*/suite-summary.json
```

### Known Limitations

- **"Cold" is process-cold, not machine-cold.** ANE kernel caches in `~/Library/Caches/` persist across runs. The cold run warms the OS cache, not the ANE cache.
- **jq required.** Aggregation needs jq. Scripts fail hard with a clear message if absent.
- **Suite runtime.** 1 cold + 9 warm invocations (~5 min at ~30s each). Acceptable for referee-grade measurement.
- **No prompt-length normalization.** Each prompt gets its own CoreML sequence length bucket via existing `nextPowerOfTwo` logic. This is intentional — it tests real behavior.

## Lane Templates

- [benchmark-referee](lanes/benchmark-referee.md)
- [incremental-decode](lanes/incremental-decode.md)
- [head-path](lanes/head-path.md)
- [bucket-io](lanes/bucket-io.md)
