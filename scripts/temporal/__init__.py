"""Temporal feature extraction package.

Recommended entrypoints (keep):
- build_S_temp.py
  Build temporal similarity matrix S_temp (JTL + traces hybrid).
- temporal_core.py
  Core trace/log -> class-index extraction + S_temp computation utilities.
- split_traces_by_service.py
  Split OTEL NDJSON file sink (all_traces.json) into per-service files.
- temporal_manifest.py
  Quick check: do we have traces + JTL files for each app?
- temporal_gate_report.py
  Quality gate: density/offdiag/intra-inter ratio + JTL hit proxy.

Diagnostics (keep, but optional):
- inspect_otel_trace_fields.py
- check_time_alignment.py
- inspect_jtl_tx_threads.py
- diagnose_temp_failure_cases.py
- inspect_jpetstore_strict_mapping_coverage.py

Compatibility shims / placeholders (keep minimal):
- split_traces_and_logs_by_service.py (shim; use split_traces_by_service.py)
- audit_plants_mapping.py (placeholder)
"""
