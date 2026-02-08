# DayTrader Temporal Modality (S_temp) — Paper Notes (Feb 2026)

## Challenge
DayTrader exposes a **nearly uniform URL path** in traces (e.g., `url.path=/daytrader/app`).
This breaks naive *path→class* trace mapping: if we only map `url.path`, most traces collapse onto a single web entrypoint (e.g., `TradeAppServlet`), which yields **very low temporal evidence density** (off-diagonal nonzeros and density stay small).

## Solution: Query-Driven Reconstruction + Saturated Action Cliques
We introduced a **Query-Driven Reconstruction** step in trace-to-class mapping:

1. Parse `url.query` and extract the **business action** (e.g., `action=quotes`, `action=marketSummary`, `action=buy`, `action=account`, …).
2. Map each action to a **domain-aligned saturated class cluster** (a small clique) that represents the action’s execution slice across tiers (Servlet/Action/EJB/Direct/JSF/Entity).
   - Implemented as `DAYTRADER_ACTION_TO_CLASSES` in `scripts/temporal/temporal_core.py`.
3. Fallback (best-effort, conservative): if `url.query` is missing, infer a small set of high-value actions from `span.name/http.route/url.path`.

This increases trace co-occurrence edges because each span now maps to **K classes**, contributing ~K·(K−1)/2 intra-service edges per trace.

## Outcome (DayTrader)
- **offdiag_nonzero:** 92 → 412
- **overall ratio (intra/inter):** 2.441 → 1.924 (still strong; remains > 1.2 gate requirement)
- Gate status: **PASS**

Interpretation: the additional 300+ edges are largely **intra-service / intra-domain**, consistent with business-domain clustering recovered from query semantics.

## Repro steps (commands)
1. Build matrix:
   - `python scripts/temporal/build_S_temp.py --system daytrader`
2. Gate audit:
   - `python scripts/temporal/temporal_gate_report.py --system daytrader --s data/processed/temporal/daytrader_S_temp.npy`

## Key implementation points
- Query parsing and action dispatch added in `resolve_class_indices()`.
- Action clusters include only classes that exist in `data/processed/fusion/daytrader_class_order.json`.
- The gate’s evidence proxy mapping for DayTrader is kept consistent in `scripts/temporal/temporal_gate_report.py`.

## Audit snapshot (Feb 3, 2026): data-cleaning fixes that affect temporal evidence
- **JPetStore (headerless JTL)**: `build_S_temp.py` JTL loader was updated to detect headerless `.jtl` exports and fall back to standard JTL fieldnames (so `label/threadName/threadIteration` are parsed correctly). This changed build-time JTL session evidence from 0 to nonzero and allowed JTL co-occurrence edges to contribute to `jpetstore_S_temp.npy`.
- **AcmeAir (parentheses normalization)**: strict label normalization in `build_S_temp.py` was updated to strip parentheses/brackets (e.g., "REST - Login (F2)" → `restloginf2`). Without this, most labels failed to match `MINIMAL_ENTRYPOINTS` and sessions degenerated to singleton classes, producing `offdiag_nonzero=0` for the JTL component.
