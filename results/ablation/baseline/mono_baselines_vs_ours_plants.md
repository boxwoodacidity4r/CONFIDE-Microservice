# Mono baselines vs ours (plants)

Generated: (no timestamp; deterministic overwrite)

| Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | pred_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Mono2Micro_Semantic | 0.5833 | 54.84 | 31.00 | 0.5080 | -0.0094 | 0.6327 | 3 | 4 | -1 | 0.0 | with_u | data/processed/fusion/plants_pred_Mono2Micro_Semantic.json |
| Bunch_MEM_Structural | 0.6012 | 61.29 | 49.00 | 0.2540 | -0.0726 | 1.0000 | 3 | 4 | -1 | 1.0 | with_u | data/processed/fusion/plants_pred_Bunch_MEM_Structural.json |
| COGCN_SimpleFusion | 0.7140 | 67.74 | 35.00 | 0.6387 | -0.0264 | 0.7143 | 3 | 4 | -1 | 0.5 | no_u | data/processed/fusion/plants_pred_COGCN_SimpleFusion.json |
| Ours_CAC_noU | 0.4094 | 45.16 | 31.00 | 0.3193 | 0.0290 | 0.6327 | 3 | 4 | -1 | - | no_u | data/processed/fusion/plants_pred_Ours_CAC_noU.json |
| Ours_CAC_withU | 0.4094 | 45.16 | 31.00 | 0.3193 | 0.0290 | 0.6327 | 3 | 4 | -1 | - | with_u | data/processed/fusion/plants_pred_Ours_CAC_withU.json |

## Notes
- Mono2Micro_Semantic/Bunch_MEM_Structural/COGCN_SimpleFusion are *equivalent reproductions* under our evidence space by switching matrix inputs (mu) and uncertainty (U).
- Ours_CAC_noU keeps the CAC pipeline identical to Ours_CAC_withU except forcing U≡0 (strict uncertainty ablation under the same mu/cap/K-lock).
- K-lock uses --target_from_gt to keep service granularity comparable.
- IFN/NED/SM/ICP are reported only when dependency matrix exists in data/processed/dependency/<system>_dependency_matrix.json.
- pred_path records the exact prediction JSON used for evaluation (artifact reproducibility).
