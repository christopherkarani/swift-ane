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
- [x] Add `.esp` manifest sections for output-head and draft metadata with exact/near-exact/approximate labeling.
- [x] Validate factored-head and draft sidecar file references during bundle/runtime open.
- [x] Expose selected output-head and draft features through runtime resolution.
- [x] Add CLI/exporter support for output-head and draft manifest metadata.
- [x] Run verification builds/tests for the output-head/draft contract slice and a real bundle smoke run.
- [x] Add real model/export/runtime execution for a cheaper factored Stories head and benchmark it on the Stories release path.
- [ ] Retain a factored Stories head only if real Stories throughput wins and prompt quality is acceptable.
- [x] Produce the first stable Stories student artifact with the distillation pipeline and verify short/long prompt quality against the retained exact bundle.
- [x] Export an exact two-token Stories draft artifact from the stable student and package it through `.esp` draft metadata.
- [x] Add real bundle/runtime execution for exact two-token Stories decoding with acceptance accounting and benchmark it on the Stories release path.
- [ ] Retain an exact two-token Stories draft path only if the real Stories release benchmark improves over the retained exact bundle.
- [x] Build and benchmark smaller teacher-truncated Stories student SKUs as the next candidate standalone or draft proposer artifacts.
- [ ] Benchmark intermediate teacher-truncated Stories students that may preserve enough quality to be viable draft proposers.
- [ ] Retain teacher-truncated pipeline support only if it continues to produce useful measured search results.
- [x] Benchmark a 4-layer teacher-truncated Stories student on short, long, and real release benchmark paths.
- [x] Train a real distilled 6-8 layer Stories student and record honest acceptance-quality metrics against the teacher.
- [ ] Replace the CPU-only exact draft proof path with an ANE-backed or hybrid draft/verifier path before the next release benchmark.
- [ ] Retain a distilled draft student only if acceptance is high enough to preserve quality while improving real Stories wall-clock throughput.

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
| Output-head/draft contract slice | `swift test --filter 'ESPBundleTests|ESPRuntimeTests|ESPConvertTests'` + release `espc`/`esprun` rebuild + `esprun`/`espresso-generate` smoke on `/tmp/stories110m-contract-proof.esp` | no bundle contract for factored-head or draft metadata | manifest/runtime/CLI support added; proof bundle packaged and generated successfully | contract-only verification; no throughput claim, retained path unchanged | keep |
| Factored Stories head `r256` | `python3 scripts/factorize_stories_output_head.py ... --rank 256` then `espresso-generate ... /tmp/stories110m-factored-r256.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `99.48 tok/s`, `compile_ms=802.03`, exact baseline text | `107.81 tok/s`, `compile_ms=695.04`, `exact_head_backend=ane_factored_classifier` | near-exact label was too optimistic; short and long prompts diverged badly | reject artifact, keep support/tooling |
| Factored Stories head `r512` | `python3 scripts/factorize_stories_output_head.py ... --rank 512` then `espresso-generate ... /tmp/stories110m-factored-r512.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `99.48 tok/s`, `compile_ms=802.03`, exact baseline text | `111.11 tok/s`, `compile_ms=819.02`, `exact_head_backend=ane_factored_classifier` | faster, but short and long prompts still diverged materially from the retained exact path | reject artifact, keep support/tooling |
| Stable student exact-copy artifact | `python3 scripts/distill_stories_native.py --config configs/stories/distill-stable-copy.json` then short/long `espresso-generate --bundle /tmp/stories110m-stable-copy.esp ...` | no stable Stories-native student artifact | `/tmp/stories110m-stable-copy.esp` exported with `initialization_mode=teacher_copy`; short and long prompt continuations matched `/tmp/stories110m-ctx256-v1.esp` exactly | exact stable student proof; not a throughput claim by itself | keep |
| Exact two-token Stories draft proof | `python3 scripts/package_stories_exact_two_token_draft.py ...` then `espresso-generate ... /tmp/stories110m-exact-two-token-draft.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `115.86 tok/s`, `compile_ms=1004.71`, `first_token_ms=1.45`, `median_token_ms=8.75`, `p95_token_ms=10.39`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | `76.16 tok/s`, `compile_ms=0.00`, `first_token_ms=1.15`, `median_token_ms=14.23`, `p95_token_ms=19.66`, `exact_head_backend=cpu_exact_two_token_draft`, `cached_bindings_enabled=false`, `committed_exact_tokens_per_pass=1.9688`, `accepted_future_tokens_per_pass=1.9688` | exact; short and long prompt continuations matched the retained exact bundle, but wall-clock throughput regressed materially | reject artifact, keep support/tooling |
| Teacher-truncated 6-layer Stories student | `python3 scripts/distill_stories_native.py --config configs/stories/distill-truncate-6layer.json` then `espresso-generate ... /tmp/stories110m-truncate-6layer.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `115.86 tok/s`, exact retained text | `179.28 tok/s`, `compile_ms=658.73`, `first_token_ms=1.38`, `median_token_ms=5.71`, `p95_token_ms=6.64`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | approximate; short and long prompts diverged badly despite large speedup | reject artifact, keep truncation pipeline support |
| Teacher-truncated 4-layer Stories student | `python3 scripts/distill_stories_native.py --config /tmp/stories110m-truncate-4layer.json` then short/long checks plus `espresso-generate ... /tmp/stories110m-truncate-4layer.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `115.86 tok/s`, exact retained text | `187.85 tok/s`, `compile_ms=702.02`, `first_token_ms=1.48`, `median_token_ms=5.44`, `p95_token_ms=6.38`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | approximate; short prompt collapsed into repeated `ely`, long prompt diverged strongly despite the speedup | reject artifact |
| Teacher-truncated 10-layer Stories student | `python3 scripts/distill_stories_native.py --config /tmp/stories110m-truncate-10layer.json` then short/long `espresso-generate --bundle /tmp/stories110m-truncate-10layer.esp ...` | 6-layer candidate showed quality cliff | short prompt `90.93 tok/s`, long prompt `130.25 tok/s` single runs | approximate; still materially divergent, and speedup signal was much weaker than 6-layer | reject artifact |
| Trained 6-layer distilled Stories student | `python3 scripts/distill_stories_native.py --config configs/stories/distill-truncate-6layer-train.json` then short/long `espresso-generate --bundle /tmp/stories110m-truncate-6layer-trained.esp ...` | zero-step 6-layer student had `teacher_token_agreement=0.1101`, `exact_two_token_token0_match_rate=0.2857`, `exact_two_token_future_accept_rate=0.0`, and unusable prompts | trained student reached `teacher_token_agreement=0.6239`, `mean_teacher_student_kl=5.80`, `exact_two_token_token0_match_rate=0.4286`, still `exact_two_token_future_accept_rate=0.0`; short prompt `100.56 tok/s` single run with `compile_ms=12135.85`, long prompt `61.40 tok/s` single run with `compile_ms=11942.05` | approximate; offline agreement improved materially, but generation quality remained poor and ANE compile instability was severe | reject artifact, keep evaluation/training support |
| Trained 8-layer distilled Stories student | `python3 scripts/distill_stories_native.py --config configs/stories/distill-truncate-8layer-train.json` then short/long `espresso-generate --bundle /tmp/stories110m-truncate-8layer-trained.esp ...` | zero-step 8-layer student had `teacher_token_agreement=0.1560`, `exact_two_token_token0_match_rate=0.2857`, `exact_two_token_future_accept_rate=0.0` | trained student reached `teacher_token_agreement=0.6881`, `mean_teacher_student_kl=3.69`, `exact_two_token_token0_match_rate=0.5714`, still `exact_two_token_future_accept_rate=0.0`; short prompt `112.39 tok/s` single run with `compile_ms=19000.51`, long prompt `116.00 tok/s` single run with `compile_ms=18848.10` | approximate; strongest agreement so far, but prompts still diverged and ANE compile instability made it unsuitable as a retained draft proposer | reject artifact, keep evaluation/training support |

## Review

- Retained:
  bundle contract v`1.1.0`, context-target packing, measured bundle token latencies, output-head/draft manifest-runtime contract support, factorized-head runtime/tooling support, the stable teacher-copied Stories student pipeline/artifact, the teacher-truncation student pipeline support, and the executed exact two-token draft bundle/runtime proof path.
- Remaining:
  a retained factored-head Stories artifact with real release-benchmark evidence and acceptable prompt quality, a retained exact two-token or multi-token Stories artifact that beats the current release benchmark, and a retained optimized Stories artifact beyond the ctx256 packaging lane.
