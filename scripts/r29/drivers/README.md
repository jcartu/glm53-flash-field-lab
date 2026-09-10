# R29-R31 execution drivers

Standalone campaign drivers copied from the local evidence root
`/home/josh/omp-workspace/drock-lmcache/r29-execution-20260909/`. They are
pinned to absolute evidence paths under that root and to pinned container
image digests, so they are not portable without editing the constants at the
top of each file.

Every driver was executed through the guarded coordinator
(`scripts/r26/run_qualification.py`), which pauses the production container,
runs the phase under `GPU_ISOLATION_MODE=strict`, and restores production
unchanged afterwards. Do not run them directly against GPUs without that
guard.

- `resume_cache_lifecycle.py` — continued the PR64-overlay disk-cache lifecycle after the observer correction.
- `r30_cache_campaign.py` — stock R30 candidate vs stock R29 negative control, lifecycle-only.
- `r30_dcp4_identity_followup.py` — stock R29 controls for the R30 DCP4 identity-gate question.
- `dflash2_c8_confirmation.py` — DFlash2 C8 verifier-step collapse: capture-size 64 vs 32.
- `r31_checkpoint_comparison.py` — R31 checkpoint comparison (nvidia/published/qad2500) plus R31 sanity arms.
- `restoration_smoke.py` — production pause/restore smoke for the coordinator.
