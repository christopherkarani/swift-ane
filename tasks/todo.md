# Stories Convert -> Optimize

## Status

- [x] Confirm current baseline on the real Stories release benchmark.
- [x] Add repo-local task tracking and keep this file current.
- [x] Expand the `.esp` manifest/runtime contract for model tier, behavior class, context target, and optimization lineage.
- [x] Replace bundle benchmark synthetic token latencies with measured per-token latencies.
- [x] Add explicit context-target Stories SKU packing and runtime resolution support.
- [x] Package and verify a first-class `stories110m-ctx256` `.esp` artifact.
- [x] Add GQA/MQA Stories variant schema/runtime compatibility coverage.
- [x] Add a reproducible distilled Stories-native pipeline with export/eval metadata.
- [ ] Add manifest/runtime support for a factored output head.
- [ ] Add manifest/runtime support for a draft or multi-token head with verifier accounting.
- [ ] Run verification builds/tests/benchmarks and record retained outcomes.

## Baseline

- Date: 2026-03-26
- Command:
  `./.build/arm64-apple-macosx/release/espresso-generate generate --bundle /tmp/stories110m.esp --max-tokens 64 --benchmark-generate --compare-warmup 1 --compare-iterations 3 Hello`
- Result:
  `tok_per_s=102.51`, `median_token_ms=9.61`, `p95_token_ms=9.75`, `first_token_ms=1.62`, `compile_ms=1043.31`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`
- Quality note:
  exact retained Stories release path; generated short story continuation without runtime failure.

## Benchmark Ledger

| Change | Command | Before | After | Quality | Decision |
| --- | --- | --- | --- | --- | --- |
| Baseline | `espresso-generate ... /tmp/stories110m.esp ... Hello` | n/a | `102.51 tok/s` | exact retained path | keep |
| Measured bundle latencies | `swift test --filter 'ESPBundleTests|ESPRuntimeTests|ESPConvertTests|EspressoGenerateTests'` + release rebuild | synthetic bundle `median/p95` | bundle path now reports measured token latencies from `GenerationResult` | exact metric fix, no decode-path change | keep |
| `stories110m-ctx256` bundle | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | `/tmp/stories110m.esp`: `98.36 tok/s`, `compile_ms=1188.32` | `104.01 tok/s`, `compile_ms=986.73` | exact; short-prompt text matched baseline | keep |
| `stories110m-ctx256` long prompt | `espresso-generate generate --bundle /tmp/stories110m-ctx256-v1.esp --max-tokens 32 --prompt 'In a distant future ...'` | baseline text matched, `106.02 tok/s` single run | text matched, `65.43 tok/s` single run | exact text match; single-run latency noisy, not used as retain gate | informational |
| Distill proof artifact | `python3 scripts/distill_stories_native.py --config configs/stories/distill-proof.json --dry-run` then `espresso-generate generate --bundle /tmp/stories110m-distill-proof.esp --max-tokens 8 Hello` | no pipeline | `.esp` proof artifact exported and ran, but `compile_ms=26778.79`, retries/failures high, text approximate/garbled | approximate proof-only, not a retained serving path | keep pipeline, reject artifact as product path |
| GQA proof artifact | `python3 scripts/distill_stories_native.py --config configs/stories/gqa4-proof.json --dry-run` then `espresso-generate generate --bundle /tmp/stories110m-gqa4-proof.esp --max-tokens 4 Hello` | no runnable Stories/GQA artifact | `.esp` GQA proof artifact exported and ran, but `compile_ms=26621.03`, retries/failures high, text approximate/garbled | approximate proof-only, validates contract/runtime compatibility, not a retained serving path | keep support, reject artifact as product path |

## Review

- Retained:
  bundle contract v`1.1.0`, context-target packing, measured bundle token latencies, and the executed distillation/export proof pipelines for baseline and GQA student artifacts.
- Remaining:
  factored-head manifest/runtime wiring, draft or multi-token manifest/runtime wiring, and a retained optimized Stories artifact beyond the ctx256 packaging lane.
