# Mono baselines vs ours (jpetstore)

Generated: (no timestamp; deterministic overwrite)

| Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | pred_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Mono2Micro_Semantic | 0.3381 | 39.73 | 49.00 | 0.1356 | 0.0156 | 0.5326 | 3 | 4 | -1 | 0.0 | with_u | data/processed/fusion/jpetstore_pred_Mono2Micro_Semantic.json |
| Bunch_MEM_Structural | 0.2953 | 27.40 | 67.00 | 0.1913 | 0.0116 | 0.7283 | 4 | 4 | +0 | 1.0 | with_u | data/processed/fusion/jpetstore_pred_Bunch_MEM_Structural.json |
| COGCN_SimpleFusion | 0.3421 | 32.88 | 42.00 | 0.2473 | 0.0159 | 0.4565 | 3 | 4 | -1 | 0.5 | no_u | data/processed/fusion/jpetstore_pred_COGCN_SimpleFusion.json |
| Ours_CAC_noU | 0.4077 | 46.58 | 40.00 | 0.3889 | 0.0215 | 0.4348 | 3 | 4 | -1 | 0.1 | no_u | data/processed/fusion/jpetstore_pred_Ours_CAC_noU.json |
| Ours_CAC_withU | 0.4318 | 46.58 | 52.00 | 0.2183 | 0.0013 | 0.5652 | 3 | 4 | -1 | 0.1 | with_u | data/processed/fusion/jpetstore_pred_Ours_CAC_withU.json |

## Notes
- Mono2Micro_Semantic/Bunch_MEM_Structural/COGCN_SimpleFusion are *equivalent reproductions* under our evidence space by switching matrix inputs (mu) and uncertainty (U).
- Ours_CAC_noU keeps the CAC pipeline identical to Ours_CAC_withU except forcing U≡0 (strict uncertainty ablation under the same mu/cap/K-lock).
- K-lock uses --target_from_gt to keep service granularity comparable.
- IFN/NED/SM/ICP are reported only when dependency matrix exists in data/processed/dependency/<system>_dependency_matrix.json.
- pred_path records the exact prediction JSON used for evaluation (artifact reproducibility).
