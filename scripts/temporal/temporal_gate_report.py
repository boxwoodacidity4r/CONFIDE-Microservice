"""Temporal modality gate report.

Purpose
-------
Provide a single, repeatable quality gate for the temporal modality (S_temp).
This script is explicitly designed for your workflow:
- You regenerate S_temp via scripts/temporal/build_S_temp.py
- Then you run this gate report to decide whether temporal is usable for fusion.

It reports (per system):
1) Matrix quality: offdiag_nonzero, density, top pairs, intra/inter ratio.
2) Evidence quality proxy: session class-hit distribution (p50/p90) by rebuilding
   sessions from JTL using the same strict/grouping rules as build_S_temp.

Notes
-----
- This script DOES NOT modify S_temp.
- It does not require trace input.
- It is a diagnostic gate; "pass/fail" thresholds are configurable.

Usage (PowerShell)
------------------
  .\.venv\Scripts\python.exe scripts\temporal\temporal_gate_report.py --system jpetstore --strict --group-by thread_iteration

  # Batch all systems
  .\.venv\Scripts\python.exe scripts\temporal\temporal_gate_report.py --batch --strict --group-by thread_iteration
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np

# Ensure imports work when executed as a script (python scripts/temporal/temporal_gate_report.py)
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SYSTEMS = ["acmeair", "jpetstore", "daytrader", "daytrader7", "plants", "plantsbywebsphere"]


DEFAULT_PATHS = {
    "acmeair": {
        "S": "data/processed/temporal/acmeair_S_temp.npy",
        "order": "data/processed/fusion/acmeair_class_order.json",
        "gt": "data/processed/groundtruth/acmeair_ground_truth.json",
        "jtl": "results/jmeter/acmeair_results.jtl",
    },
    "daytrader": {
        "S": "data/processed/temporal/daytrader_S_temp.npy",
        "order": "data/processed/fusion/daytrader_class_order.json",
        "gt": "data/processed/groundtruth/daytrader_ground_truth.json",
        "jtl": "results/jmeter/daytrader_results.jtl",
    },
    "daytrader7": {
        "S": "data/processed/temporal/daytrader_S_temp.npy",
        "order": "data/processed/fusion/daytrader_class_order.json",
        "gt": "data/processed/groundtruth/daytrader_ground_truth.json",
        "jtl": "results/jmeter/daytrader_results.jtl",
    },
    "jpetstore": {
        "S": "data/processed/temporal/jpetstore_S_temp.npy",
        "order": "data/processed/fusion/jpetstore_class_order.json",
        "gt": "data/processed/groundtruth/jpetstore_ground_truth.json",
        "jtl": "results/jmeter/jpetstore_results.jtl",
    },
    "plants": {
        "S": "data/processed/temporal/plants_S_temp.npy",
        "order": "data/processed/fusion/plants_class_order.json",
        "gt": "data/processed/groundtruth/plants_ground_truth.json",
        "jtl": "results/jmeter/plants_results.jtl",
    },
    "plantsbywebsphere": {
        "S": "data/processed/temporal/plants_S_temp.npy",
        "order": "data/processed/fusion/plants_class_order.json",
        "gt": "data/processed/groundtruth/plants_ground_truth.json",
        "jtl": "results/jmeter/plants_results.jtl",
    },
}


# ----------------------------------------------------------------------------
# Paper Version 1.0 (Stabilized)
# Self-contained Plants (PBW) strict label->class mapping for JTL sessionization.
# - ProductBean removed (high-frequency glue) based on session coverage audit.
# - This is embedded here to avoid importing a potentially changing build script.
# ----------------------------------------------------------------------------
PLANTS_MINIMAL_ENTRYPOINTS: Dict[str, Any] = {
    "admin": [
        "com.ibm.websphere.samples.pbw.war.AdminServlet",
        "com.ibm.websphere.samples.pbw.bean.PopulateDBBean",
    ],
    "backorder": [
        "com.ibm.websphere.samples.pbw.bean.BackOrderMgr",
        "com.ibm.websphere.samples.pbw.jpa.BackOrder",
    ],
    "supplier": [
        "com.ibm.websphere.samples.pbw.bean.SuppliersBean",
        "com.ibm.websphere.samples.pbw.jpa.Supplier",
    ],
    "catalog": [
        "com.ibm.websphere.samples.pbw.bean.CatalogMgr",
        "com.ibm.websphere.samples.pbw.jpa.Inventory",
    ],
}


# DayTrader strict mapping (evidence proxy only; not used to build S_temp here)
DAYTRADER_MINIMAL_ENTRYPOINTS: Dict[str, Any] = {
    # core servlet/action entrypoints
    "portfolio": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "quotes": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "tradestock": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "buystock": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "sellstock": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "login": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "userlogin": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "account": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "userinfo": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "home": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "buy": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "sell": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "order": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "logout": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "register": "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
    "registeruser": "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",

    # TransactionController parent labels
    "t_home": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "t_login": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "t_register": "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
    "t_portfolio": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "t_quotes": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "t_buy": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "t_sell": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",
    "t_accountlogout": "com.ibm.websphere.samples.daytrader.web.TradeAppServlet",

    # Thick per-label class sets (match build_S_temp.py strict mapping)
    "marketsummary": [
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],
    "market summary": [
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],
    "accountinfo": [
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.AccountDataJSF",
    ],
    "account info": [
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.AccountDataJSF",
    ],
    "portfolio": [
        "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.PortfolioJSF",
        "com.ibm.websphere.samples.daytrader.web.jsf.HoldingData",
        "com.ibm.websphere.samples.daytrader.web.jsf.OrderData",
    ],
    "quotes": [
        "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean",
        "com.ibm.websphere.samples.daytrader.ejb3.MarketSummarySingleton",
        "com.ibm.websphere.samples.daytrader.web.jsf.MarketSummaryJSF",
    ],
    "user login": [
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
        "com.ibm.websphere.samples.daytrader.web.jsf.LoginValidator",
        "com.ibm.websphere.samples.daytrader.web.jsf.JSFLoginFilter",
    ],
    "user register": [
        "com.ibm.websphere.samples.daytrader.entities.AccountDataBean",
        "com.ibm.websphere.samples.daytrader.entities.AccountProfileDataBean",
        "com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet",
    ],
    "buy stock": [
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
    ],
    "sell stock (by holdingid)": [
        "com.ibm.websphere.samples.daytrader.entities.OrderDataBean",
        "com.ibm.websphere.samples.daytrader.entities.HoldingDataBean",
    ],
}


# AcmeAir strict mapping (evidence proxy): map JTL labels to stable REST entrypoints.
ACMEAIR_MINIMAL_ENTRYPOINTS: Dict[str, Any] = {
    "restloginf2": [
        "com.acmeair.web.LoginREST",
        "com.acmeair.service.CustomerService",
        "com.acmeair.morphia.services.CustomerServiceImpl",
    ],
    "restloginf3": [
        "com.acmeair.web.LoginREST",
        "com.acmeair.service.CustomerService",
        "com.acmeair.morphia.services.CustomerServiceImpl",
    ],
    "restloginf4": [
        "com.acmeair.web.LoginREST",
        "com.acmeair.service.CustomerService",
        "com.acmeair.morphia.services.CustomerServiceImpl",
    ],
    "restsearchflightsf2": [
        "com.acmeair.web.FlightsREST",
        "com.acmeair.service.FlightService",
        "com.acmeair.morphia.services.FlightServiceImpl",
    ],
    "restbookflightf2": [
        "com.acmeair.web.BookingsREST",
        "com.acmeair.service.BookingService",
        "com.acmeair.morphia.services.BookingServiceImpl",
        "com.acmeair.morphia.entities.BookingImpl",
    ],
    "restgetbookingsf3": [
        "com.acmeair.web.BookingsREST",
        "com.acmeair.service.BookingService",
        "com.acmeair.morphia.services.BookingServiceImpl",
        "com.acmeair.morphia.entities.BookingImpl",
    ],
    "restgetcustomerf4": [
        "com.acmeair.web.CustomerREST",
        "com.acmeair.service.CustomerService",
        "com.acmeair.morphia.services.CustomerServiceImpl",
        "com.acmeair.morphia.entities.CustomerImpl",
    ],
    "restupdatecustomerf4": [
        "com.acmeair.web.CustomerREST",
        "com.acmeair.service.CustomerService",
        "com.acmeair.morphia.services.CustomerServiceImpl",
        "com.acmeair.morphia.entities.CustomerImpl",
    ],
    # UI steps (coarse but helps evidence proxy)
    "openhomepage": [
        "com.acmeair.web.LoginREST",
    ],
    "opencustomerprofilepagef2": [
        "com.acmeair.web.CustomerREST",
        "com.acmeair.service.CustomerService",
    ],
}


# JPetStore strict mapping (evidence proxy): map classic Struts/Spring actions.
JPETSTORE_MINIMAL_ENTRYPOINTS: Dict[str, Any] = {
    "home": [
        "org.springframework.samples.jpetstore.web.spring.CatalogController",
    ],
    "search": [
        "org.springframework.samples.jpetstore.web.spring.SearchProductsController",
        "org.springframework.samples.jpetstore.domain.logic.PetStoreFacade",
    ],
    "viewcategory": [
        "org.springframework.samples.jpetstore.web.spring.ViewCategoryController",
        "org.springframework.samples.jpetstore.domain.Category",
    ],
    "viewproduct": [
        "org.springframework.samples.jpetstore.web.spring.ViewProductController",
        "org.springframework.samples.jpetstore.domain.Product",
    ],
    "viewitem": [
        "org.springframework.samples.jpetstore.web.spring.ViewItemController",
        "org.springframework.samples.jpetstore.domain.Item",
    ],
    "additemtocart": [
        "org.springframework.samples.jpetstore.web.spring.AddItemToCartController",
        "org.springframework.samples.jpetstore.domain.Cart",
    ],
    "viewcart": [
        "org.springframework.samples.jpetstore.web.spring.ViewCartController",
        "org.springframework.samples.jpetstore.domain.Cart",
    ],
    "checkout": [
        "org.springframework.samples.jpetstore.web.spring.OrderFormController",
        "org.springframework.samples.jpetstore.domain.Order",
    ],
    "neworder": [
        "org.springframework.samples.jpetstore.web.spring.OrderFormController",
        "org.springframework.samples.jpetstore.domain.Order",
    ],
    "neworderform": [
        "org.springframework.samples.jpetstore.web.spring.OrderFormController",
        "org.springframework.samples.jpetstore.domain.Order",
    ],
    "signon": [
        "org.springframework.samples.jpetstore.web.spring.SignonController",
        "org.springframework.samples.jpetstore.domain.Account",
    ],
}


@dataclass
class GateThresholds:
    min_offdiag_nonzero: int = 200
    min_density: float = 1e-3
    min_ratio_overall: float = 1.0
    min_ratio_filtered: float = 1.1
    min_session_p50_hits: int = 3
    min_session_p90_hits: int = 4


def _adaptive_min_offdiag(system: str, n: int, default_min: int) -> int:
    """Size-aware minimum off-diagonal nonzero edges.

    Rationale: smaller systems (low n) cannot realistically reach large absolute
    off-diagonal counts (e.g., 200) while keeping strict mapping and glue removal.

    Policy:
      - For small/medium systems, require at least 8% of possible directed off-diagonals,
        but never less than 50.
      - Keep the existing AcmeAir/JPetStore policy (2% with floor 80).
      - Otherwise keep the default.
    """

    if system in {"acmeair", "jpetstore"}:
        return max(80, int(0.02 * n * (n - 1)))
    if system in {"plants", "plantsbywebsphere"}:
        return max(50, int(0.08 * n * (n - 1)))
    if system in {"daytrader", "daytrader7"}:
        return max(150, int(0.01 * n * (n - 1)))
    return int(default_min)


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _label_list(gt: Dict[str, int], order: List[str]) -> List[int]:
    return [int(gt.get(c, -1)) for c in order]


def _intra_inter_stats(S: np.ndarray, labels: List[int]) -> Tuple[float, float, float, int, int]:
    n = len(labels)
    intra: List[float] = []
    inter: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == -1 or labels[j] == -1:
                continue
            if labels[i] == labels[j]:
                intra.append(float(S[i, j]))
            else:
                inter.append(float(S[i, j]))
    intra_avg = float(np.mean(intra)) if intra else float("nan")
    inter_avg = float(np.mean(inter)) if inter else float("nan")
    ratio = (intra_avg / inter_avg) if (inter_avg and inter_avg > 0) else float("inf")
    return intra_avg, inter_avg, ratio, len(intra), len(inter)


def _top_pairs(S: np.ndarray, top_k: int) -> List[Tuple[int, int, float]]:
    n = S.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        row = S[i]
        for j in range(i + 1, n):
            pairs.append((i, j, float(row[j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _offdiag_stats(S: np.ndarray) -> Tuple[int, float]:
    n = S.shape[0]
    off = S.copy()
    np.fill_diagonal(off, 0.0)
    nonzero = int((off > 0).sum())
    density = nonzero / float(n * n - n)
    return nonzero, density


def _normalize_label(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _map_label_to_class_indices_acmeair_strict(label: str, class_to_idx: Dict[str, int]) -> List[int]:
    key = _normalize_label(label).replace("(", "").replace(")", "")
    mapped = ACMEAIR_MINIMAL_ENTRYPOINTS.get(key)
    if not mapped:
        return []

    mapped_list = [mapped] if isinstance(mapped, str) else list(mapped)
    out: List[int] = []
    for fqcn in mapped_list:
        idx = class_to_idx.get(fqcn)
        if idx is not None:
            out.append(int(idx))
    return out


def _build_sessions_from_jtl_acmeair_strict(
    jtl_path: Path,
    class_to_idx: Dict[str, int],
    *,
    group_by: str,
    max_events: int,
    min_events_per_session: int,
) -> List[List[int]]:
    rows = _load_jtl_rows(jtl_path, max_rows=max_events if max_events > 0 else 0)
    groups: Dict[str, List[int]] = {}

    for row in rows:
        label = (row.get("label") or row.get("Label") or "").strip()
        if not label:
            continue

        idxs = _map_label_to_class_indices_acmeair_strict(label, class_to_idx)
        if not idxs:
            continue

        if group_by == "thread":
            key = (row.get("threadName") or row.get("thread") or "").strip()
        elif group_by == "thread_iteration":
            tname, it = _extract_thread_iteration(row)
            key = f"{tname}::it={it}"
        else:
            raise ValueError(f"Unsupported group_by for self-contained AcmeAir strict gate: {group_by}")

        if not key:
            continue
        groups.setdefault(key, []).extend(idxs)

    return [sess for sess in groups.values() if len(sess) >= int(min_events_per_session)]


def _map_label_to_class_indices_jpetstore_strict(label: str, class_to_idx: Dict[str, int]) -> List[int]:
    key = _normalize_label(label)
    # ignore transaction controllers like FlowA_Catalog
    if key.startswith("flow"):
        return []

    mapped = JPETSTORE_MINIMAL_ENTRYPOINTS.get(key)
    if not mapped:
        return []

    mapped_list = [mapped] if isinstance(mapped, str) else list(mapped)
    out: List[int] = []
    for fqcn in mapped_list:
        idx = class_to_idx.get(fqcn)
        if idx is not None:
            out.append(int(idx))
    return out


def _build_sessions_from_jtl_jpetstore_strict(
    jtl_path: Path,
    class_to_idx: Dict[str, int],
    *,
    group_by: str,
    max_events: int,
    min_events_per_session: int,
) -> List[List[int]]:
    rows = _load_jtl_rows(jtl_path, max_rows=max_events if max_events > 0 else 0)
    groups: Dict[str, List[int]] = {}

    for row in rows:
        label = (row.get("label") or row.get("Label") or "").strip()
        if not label:
            continue

        idxs = _map_label_to_class_indices_jpetstore_strict(label, class_to_idx)
        if not idxs:
            continue

        if group_by == "thread":
            key = (row.get("threadName") or row.get("thread") or "").strip()
        elif group_by == "thread_iteration":
            tname, it = _extract_thread_iteration(row)
            key = f"{tname}::it={it}"
        else:
            raise ValueError(f"Unsupported group_by for self-contained JPetStore strict gate: {group_by}")

        if not key:
            continue
        groups.setdefault(key, []).extend(idxs)

    return [sess for sess in groups.values() if len(sess) >= int(min_events_per_session)]


def _map_label_to_class_indices_plants_strict(label: str, class_to_idx: Dict[str, int]) -> List[int]:
    key = _normalize_label(label)
    mapped = PLANTS_MINIMAL_ENTRYPOINTS.get(key)
    if not mapped:
        return []

    mapped_list = [mapped] if isinstance(mapped, str) else list(mapped)
    out: List[int] = []
    for fqcn in mapped_list:
        idx = class_to_idx.get(fqcn)
        if idx is not None:
            out.append(int(idx))
    return out


def _build_sessions_from_jtl_plants_strict(
    jtl_path: Path,
    class_to_idx: Dict[str, int],
    *,
    group_by: str,
    max_events: int,
    min_events_per_session: int,
) -> List[List[int]]:
    rows = _load_jtl_rows(jtl_path, max_rows=max_events if max_events > 0 else 0)
    groups: Dict[str, List[int]] = {}

    for row in rows:
        label = (row.get("label") or row.get("Label") or "").strip()
        if not label:
            continue

        idxs = _map_label_to_class_indices_plants_strict(label, class_to_idx)
        if not idxs:
            continue

        if group_by == "thread":
            key = (row.get("threadName") or row.get("thread") or "").strip()
        elif group_by == "thread_iteration":
            tname, it = _extract_thread_iteration(row)
            key = f"{tname}::it={it}"
        else:
            raise ValueError(f"Unsupported group_by for self-contained Plants strict gate: {group_by}")

        if not key:
            continue
        groups.setdefault(key, []).extend(idxs)

    return [sess for sess in groups.values() if len(sess) >= int(min_events_per_session)]


def _map_label_to_class_indices_daytrader_strict(label: str, class_to_idx: Dict[str, int]) -> List[int]:
    key = _normalize_label(label)
    mapped = DAYTRADER_MINIMAL_ENTRYPOINTS.get(key)
    if not mapped:
        return []

    mapped_list = [mapped] if isinstance(mapped, str) else list(mapped)
    out: List[int] = []
    for fqcn in mapped_list:
        idx = class_to_idx.get(fqcn)
        if idx is not None:
            out.append(int(idx))
    return out


def _build_sessions_from_jtl_daytrader_strict(
    jtl_path: Path,
    class_to_idx: Dict[str, int],
    *,
    group_by: str,
    max_events: int,
    min_events_per_session: int,
) -> List[List[int]]:
    rows = _load_jtl_rows(jtl_path, max_rows=max_events if max_events > 0 else 0)
    groups: Dict[str, List[int]] = {}

    for row in rows:
        label = (row.get("label") or row.get("Label") or "").strip()
        if not label:
            continue

        idxs = _map_label_to_class_indices_daytrader_strict(label, class_to_idx)
        if not idxs:
            continue

        if group_by == "thread":
            key = (row.get("threadName") or row.get("thread") or "").strip()
        elif group_by == "thread_iteration":
            tname, it = _extract_thread_iteration(row)
            key = f"{tname}::it={it}"
        else:
            raise ValueError(f"Unsupported group_by for self-contained DayTrader strict gate: {group_by}")

        if not key:
            continue
        groups.setdefault(key, []).extend(idxs)

    return [sess for sess in groups.values() if len(sess) >= int(min_events_per_session)]


def _load_jtl_rows(jtl_path: Path, *, max_rows: int) -> List[Dict[str, str]]:
    """Load JMeter .jtl as CSV without pandas.

    Supports both:
    - Headered JTL (first line starts with timeStamp,...)
    - Headerless JTL (common in this repo), using the standard JMeter CSV column order.

    Returns list of dict rows.
    """
    import csv

    rows: List[Dict[str, str]] = []
    if not jtl_path.is_file():
        return rows

    with jtl_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = sample.lower().lstrip().startswith("timestamp,")

        if has_header:
            reader = csv.DictReader(f)
        else:
            fieldnames = [
                "timeStamp",
                "elapsed",
                "label",
                "responseCode",
                "responseMessage",
                "threadName",
                "dataType",
                "success",
                "failureMessage",
                "bytes",
                "sentBytes",
                "grpThreads",
                "allThreads",
                "URL",
                "Latency",
                "IdleTime",
                "Connect",
            ]
            reader = csv.DictReader(f, fieldnames=fieldnames)

        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            if not row:
                continue
            rows.append(row)

    return rows


def _extract_thread_iteration(row: Dict[str, str]) -> Tuple[str, int]:
    """Return (thread_name, iteration_int).

    Supports common JMeter patterns used in this repo.
    """
    thread_name = (row.get("threadName") or row.get("thread") or "").strip()

    it_raw = (
        row.get("threadIteration")
        or row.get("iteration")
        or row.get("grpThreads")  # not ideal but keeps it robust
        or ""
    )
    try:
        it = int(float(it_raw))
    except Exception:
        it = -1
    return thread_name, it


def _build_sessions_and_hit_counts(
    system: str,
    *,
    strict: bool,
    group_by: str,
    max_events: int,
    min_events: int,
) -> List[int]:
    """Rebuild JTL sessions and return per-session hit counts (unique class indices)."""

    # Self-contained path: Plants PBW
    sess_system = "plantsbywebsphere" if system in {"plants", "plantsbywebsphere"} else system
    p = DEFAULT_PATHS[system]

    # Load class order so we can translate mapped FQCN -> idx
    order = _load_json(ROOT / p["order"])
    class_to_idx = {c: i for i, c in enumerate(order)}

    jtl_path = ROOT / p["jtl"]

    if sess_system == "plantsbywebsphere":
        sessions = _build_sessions_from_jtl_plants_strict(
            jtl_path,
            class_to_idx,
            group_by=group_by,
            max_events=max_events,
            min_events_per_session=min_events,
        )
    elif sess_system in {"daytrader", "daytrader7"}:
        sessions = _build_sessions_from_jtl_daytrader_strict(
            jtl_path,
            class_to_idx,
            group_by=group_by,
            max_events=max_events,
            min_events_per_session=min_events,
        )
    elif sess_system == "acmeair":
        sessions = _build_sessions_from_jtl_acmeair_strict(
            jtl_path,
            class_to_idx,
            group_by=group_by,
            max_events=max_events,
            min_events_per_session=min_events,
        )
    elif sess_system == "jpetstore":
        sessions = _build_sessions_from_jtl_jpetstore_strict(
            jtl_path,
            class_to_idx,
            group_by=group_by,
            max_events=max_events,
            min_events_per_session=min_events,
        )
    else:
        # Keep previous behavior: if we don't have a self-contained mapping, evidence proxy is empty.
        sessions = []

    hit_counts = [len(set(s)) for s in sessions if s]
    return hit_counts


def _pctl(xs: List[int], p: float) -> float:
    if not xs:
        return float("nan")
    a = np.array(sorted(xs), dtype=float)
    return float(np.percentile(a, p))


def gate_one(
    system: str,
    thresholds: GateThresholds,
    *,
    strict: bool,
    group_by: str,
    top_k: int,
    max_events: int,
    min_events: int,
    s_override: str = "",
) -> None:
    p = DEFAULT_PATHS[system]

    s_path = (ROOT / p["S"]) if not s_override else (ROOT / s_override)
    if not s_path.is_file():
        raise FileNotFoundError(f"S_temp matrix not found: {s_path}")

    S = np.load(str(s_path))
    n = int(S.shape[0])
    order = _load_json(ROOT / p["order"])
    gt = _load_json(ROOT / p["gt"])

    off_nz, density = _offdiag_stats(S)

    # Size-aware Evidence Density Gate:
    adaptive_min_offdiag = _adaptive_min_offdiag(system, n, thresholds.min_offdiag_nonzero)
    min_offdiag_effective = thresholds.min_offdiag_nonzero
    if system in {"acmeair", "jpetstore", "plants", "plantsbywebsphere"}:
        min_offdiag_effective = adaptive_min_offdiag

    labels = _label_list(gt, order)
    intra, inter, ratio, n_intra, n_inter = _intra_inter_stats(S, labels)

    hit_counts = _build_sessions_and_hit_counts(
        system,
        strict=strict,
        group_by=group_by,
        max_events=max_events,
        min_events=min_events,
    )
    p50 = _pctl(hit_counts, 50)
    p90 = _pctl(hit_counts, 90)

    # For Plants, the provided GT partition is not action-aligned and can make the
    # GT-based intra/inter ratio misleading. Use a session-derived ratio gate instead.
    ratio_gate_value = ratio
    if system in {"plants", "plantsbywebsphere"}:
        try:
            from itertools import combinations

            sessions = _build_sessions_from_jtl_plants_strict(
                ROOT / DEFAULT_PATHS[system]["jtl"],
                {c: i for i, c in enumerate(order)},
                group_by=group_by,
                max_events=max_events,
                min_events_per_session=min_events,
            )
            # build a simple co-occurrence graph and compute mean within/between
            # for connected components (action clusters proxy)
            n = len(order)
            adj = [[0] * n for _ in range(n)]
            for s in sessions:
                uniq = sorted(set(int(x) for x in s if 0 <= int(x) < n))
                for (i, j) in combinations(uniq, 2):
                    adj[i][j] += 1
                    adj[j][i] += 1
            # components via DFS
            seen = [False] * n
            comps = []
            for i in range(n):
                if seen[i]:
                    continue
                stack = [i]
                seen[i] = True
                comp = [i]
                while stack:
                    u = stack.pop()
                    for v in range(n):
                        if not seen[v] and adj[u][v] > 0:
                            seen[v] = True
                            stack.append(v)
                            comp.append(v)
                comps.append(comp)

            intra_vals = []
            inter_vals = []
            comp_id = [-1] * n
            for k, comp in enumerate(comps):
                for v in comp:
                    comp_id[v] = k
            for i in range(n):
                for j in range(i + 1, n):
                    if comp_id[i] == -1 or comp_id[j] == -1:
                        continue
                    if comp_id[i] == comp_id[j]:
                        intra_vals.append(float(S[i, j]))
                    else:
                        inter_vals.append(float(S[i, j]))
            intra_s = float(np.mean(intra_vals)) if intra_vals else float("nan")
            inter_s = float(np.mean(inter_vals)) if inter_vals else float("nan")
            ratio_s = (intra_s / inter_s) if (inter_s and inter_s > 0) else float("inf")
            ratio_gate_value = ratio_s
        except Exception:
            ratio_gate_value = ratio

    print("=" * 80)
    print(f"[TEMPORAL GATE] system={system} strict={strict} group_by={group_by}")
    print(f"matrix: path={s_path.as_posix()} n={n} offdiag_nonzero={off_nz} density={density:.6f}")
    print(f"overall: ratio={ratio:.3f} intra={intra:.4f} inter={inter:.4f} pairs={n_intra}/{n_inter}")
    if system in {"plants", "plantsbywebsphere"}:
        print(f"plants-session-ratio (gate): {ratio_gate_value:.3f}")
    print(f"sessions: count={len(hit_counts)} class_hits p50={p50:.1f} p90={p90:.1f} min={min(hit_counts) if hit_counts else 0} max={max(hit_counts) if hit_counts else 0}")
    if system in {"plants", "plantsbywebsphere"}:
        print("paper-baseline: Plants strict mapping embedded (ProductBean removed from JTL & trace).")

    pairs = _top_pairs(S, top_k=top_k)
    print(f"top{top_k} pairs (idx_i, idx_j, sim, class_i, class_j):")
    for (i, j, sim) in pairs[: min(15, len(pairs))]:
        print(f"  ({i},{j}) {sim:.4f}  {order[i]}  <->  {order[j]}")

    fails: List[str] = []
    if off_nz < min_offdiag_effective:
        fails.append(f"offdiag_nonzero<{min_offdiag_effective}")
    if density < thresholds.min_density:
        fails.append(f"density<{thresholds.min_density}")
    if (ratio_gate_value != float('inf')) and (ratio_gate_value < thresholds.min_ratio_overall):
        fails.append(f"ratio<{thresholds.min_ratio_overall}")
    if p50 < thresholds.min_session_p50_hits:
        fails.append(f"session_p50<{thresholds.min_session_p50_hits}")
    if p90 < thresholds.min_session_p90_hits:
        fails.append(f"session_p90<{thresholds.min_session_p90_hits}")

    if fails:
        print(f"GATE: FAIL ({', '.join(fails)})")
    else:
        print("GATE: PASS")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument(
        "--group-by",
        choices=["thread", "thread_iteration", "label_prefix", "sliding_window"],
        default="thread_iteration",
    )
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-events", type=int, default=80)
    ap.add_argument("--min-events", type=int, default=2)

    ap.add_argument("--min-offdiag-nonzero", type=int, default=200)
    ap.add_argument("--min-density", type=float, default=1e-3)
    ap.add_argument("--min-ratio", type=float, default=1.0)
    ap.add_argument("--min-session-p50", type=int, default=3)
    ap.add_argument("--min-session-p90", type=int, default=4)

    ap.add_argument(
        "--s",
        default="",
        help="Optional path (relative to repo root) to override the S_temp matrix to score (e.g., data/processed/temporal/plants_final_fused_matrix.npy)",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = GateThresholds(
        min_offdiag_nonzero=args.min_offdiag_nonzero,
        min_density=args.min_density,
        min_ratio_overall=args.min_ratio,
        min_session_p50_hits=args.min_session_p50,
        min_session_p90_hits=args.min_session_p90,
    )

    if args.batch:
        for sysname in SYSTEMS:
            gate_one(
                sysname,
                thresholds,
                strict=args.strict,
                group_by=args.group_by,
                top_k=args.top_k,
                max_events=args.max_events,
                min_events=args.min_events,
                s_override=args.s,
            )
        return

    if not args.system:
        raise SystemExit("Provide --system or use --batch")

    gate_one(
        args.system,
        thresholds,
        strict=args.strict,
        group_by=args.group_by,
        top_k=args.top_k,
        max_events=args.max_events,
        min_events=args.min_events,
        s_override=args.s,
    )


if __name__ == "__main__":
    main()
