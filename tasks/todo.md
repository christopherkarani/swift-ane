# TODO

- [x] Audit local artifact/model availability and confirm that the branch still lacks a principled recurrent-student training/export path.
- [x] Add failing tests for a checkpoint-to-generation-model export seam and an offline exact-acceptance evaluator seam.
- [x] Implement a local real-text corpus builder plus export path so this repo can produce a real local-data teacher artifact without external downloads.
- [x] Implement a CPU/offline recurrent-student path for the local-data bigram contract: recurrent forward model, export to `RecurrentGenerationWeightStore`, and a two-step future-head sidecar.
- [x] Run the offline gate on the real local artifact and report parity, committed exact tokens/pass, and accepted future tokens/pass before any ANE benchmark.
- [x] Rerun `scripts/reproduce_exact_4x.sh` with `INPUT_MODE=recurrent-checkpoint`, a real recurrent artifact, a real future sidecar, and a matching zero-weight CoreML trunk in the same session.
- [x] Package the result like a claim: raw JSON, artifact hashes, exact commands, docs update, Wax notes, handoff, and flush.

# Review

- The branch now has a recoverable real local-data artifact route: `espresso-train` can build a local text dataset, export a deterministic bigram teacher generation model, export matching recurrent weights plus a `t+2` future sidecar, and write an offline-gate JSON summary.
- The offline gate on the saved artifact is strong on its own contract: prompt token `35`, parity `match`, `committed_exact_tokens/pass = 2.0`, `accepted_future_tokens/pass = 1.0`.
- The matched public harness on the saved real artifact is not a 4x story. Across `5` runs, the exact two-step path reached `2.2881979166666664 ms/token` vs control `2.3190312500000001 ms/token` and matching zero-weight CoreML `5.049015625 ms/token`, only about `2.214x` over CoreML.
- Interpretation: the exact multi-token mechanism survives off the synthetic echo path, but the large speedup does not. This closes the “real local artifact through the same harness” question as a strong negative for any public `4x` claim on this artifact family.
