"""Rebuild groundtruth for monolith datasets using IBM microservice repos as authority.

This workspace contains:
- Monolith class lists in `data/processed/fusion/{system}_class_order.json`
- Existing groundtruth maps in `data/processed/groundtruth/*_ground_truth.json`
- IBM microservice reference repos in `data/raw/` (used as *architectural authority*)

Goal
----
Generate updated groundtruth maps + meta reports that are:
- Coverage-complete for classes present in class_order (minimize -1)
- Cluster-balanced (avoid a single giant cluster)
- Defensible: every heuristic mapping is explicitly recorded in *_meta.json

Currently implemented:
- acmeair: interface->impl propagation for morphia/wxs + entity assignment per service
- daytrader: entity-centered mapping aligned with the DayTrader microservices split

Usage
-----
python scripts/groundtruth/rebuild_groundtruth_from_ibm_microservices.py --system acmeair
python scripts/groundtruth/rebuild_groundtruth_from_ibm_microservices.py --system daytrader
python scripts/groundtruth/rebuild_groundtruth_from_ibm_microservices.py --batch

"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def _load_class_order(system: str) -> List[str]:
    p = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    return _read_json(p)


def _load_groundtruth(system: str) -> Dict[str, int]:
    p = ROOT / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
    return _read_json(p)


def _summarize_gt(gt: Dict[str, int], *, class_order: List[str]) -> Dict[str, Any]:
    present = set(class_order)
    labeled = {c: k for c, k in gt.items() if c in present and k != -1}
    unlabeled = [c for c in class_order if gt.get(c, -1) == -1]
    counts = Counter(labeled.values())

    # Business Context Coverage: treat k!=-1 as business-context labeled (active clusters)
    business_labeled = len(labeled)
    infra_labeled = len(unlabeled)
    total = len(class_order)

    return {
        "class_count": total,
        "covered_labeled": business_labeled,
        "covered_unlabeled": infra_labeled,
        "coverage_ratio": (business_labeled / total) if total else 0.0,
        "business_context_coverage": {
            "business_labeled": business_labeled,
            "infra_or_excluded": infra_labeled,
            "business_ratio": (business_labeled / total) if total else 0.0,
            "infra_ratio": (infra_labeled / total) if total else 0.0,
        },
        "cluster_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "largest_cluster": max(counts.values()) if counts else 0,
    }


@dataclass
class GTResult:
    gt: Dict[str, int]
    meta: Dict[str, Any]


def rebuild_acmeair() -> GTResult:
    system = "acmeair"
    class_order = _load_class_order(system)
    old = _load_groundtruth(system)

    # Define 4 business clusters + infra(-1)
    # clusters:
    # 0 customers, 1 booking, 2 flight, 3 auth/session
    CLUSTERS = {
        "customerservice": 0,
        "bookingservice": 1,
        "flightservice": 2,
        "authservice": 3,
    }

    # Start from old mapping (keeps any manual curation)
    gt = {c: int(old.get(c, -1)) for c in class_order}

    # Seed rules (deterministic)
    seeds: Dict[str, int] = {
        # REST edges
        "com.acmeair.web.CustomerREST": CLUSTERS["customerservice"],
        "com.acmeair.web.BookingsREST": CLUSTERS["bookingservice"],
        "com.acmeair.web.FlightsREST": CLUSTERS["flightservice"],
        "com.acmeair.web.LoginREST": CLUSTERS["authservice"],
        "com.acmeair.web.RESTCookieSessionFilter": CLUSTERS["authservice"],
        # service interfaces
        "com.acmeair.service.CustomerService": CLUSTERS["customerservice"],
        "com.acmeair.service.BookingService": CLUSTERS["bookingservice"],
        "com.acmeair.service.TransactionService": CLUSTERS["bookingservice"],
        "com.acmeair.service.FlightService": CLUSTERS["flightservice"],
        # entities by name
        "com.acmeair.entities.Customer": CLUSTERS["customerservice"],
        "com.acmeair.entities.CustomerAddress": CLUSTERS["customerservice"],
        "com.acmeair.entities.CustomerSession": CLUSTERS["authservice"],
        "com.acmeair.entities.Booking": CLUSTERS["bookingservice"],
        "com.acmeair.entities.BookingPK": CLUSTERS["bookingservice"],
        "com.acmeair.entities.Flight": CLUSTERS["flightservice"],
        "com.acmeair.entities.FlightSegment": CLUSTERS["flightservice"],
        "com.acmeair.entities.AirportCodeMapping": CLUSTERS["flightservice"],
        # loaders
        "com.acmeair.loader.CustomerLoader": CLUSTERS["customerservice"],
        "com.acmeair.loader.FlightLoader": CLUSTERS["flightservice"],
    }

    applied: List[Dict[str, Any]] = []
    forced_entity_mapping: List[str] = []
    interface_impl_propagation: List[Dict[str, Any]] = []

    for c, k in seeds.items():
        if c in gt:
            prev = gt[c]
            gt[c] = k
            applied.append({"class": c, "from": prev, "to": k, "rule": "seed"})

    # Interface -> impl propagation (morphia/wxs)
    service_to_cluster = {
        "CustomerService": CLUSTERS["customerservice"],
        "BookingService": CLUSTERS["bookingservice"],
        "FlightService": CLUSTERS["flightservice"],
    }

    for c in class_order:
        # Map service impls
        for svc, k in service_to_cluster.items():
            if c.endswith(f".{svc}Impl") and (
                c.startswith("com.acmeair.morphia.services.")
                or c.startswith("com.acmeair.wxs.service.")
            ):
                prev = gt[c]
                gt[c] = k
                interface_impl_propagation.append(
                    {"class": c, "from": prev, "to": k, "rule": f"interface->impl:{svc}"}
                )

        # Map entity impls (Morphia/WXS layers)
        # NOTE: entity impl class names are strongly aligned with entity names.
        if c.startswith("com.acmeair.morphia.entities.") or c.startswith("com.acmeair.wxs.entities."):
            base = c.rsplit(".", 1)[-1]
            # normalize: CustomerImpl -> Customer
            if base.endswith("Impl"):
                base_entity = base[: -len("Impl")]
                # BookingPKImpl -> BookingPK
                # FlightPKImpl exists only in WXS list (class_order)
                # We assign by entity prefix.
                target = None
                if base_entity.startswith("Customer"):
                    target = CLUSTERS["customerservice"] if base_entity != "CustomerSession" else CLUSTERS["authservice"]
                elif base_entity.startswith("Booking"):
                    target = CLUSTERS["bookingservice"]
                elif base_entity.startswith("Flight") or base_entity.startswith("Airport"):
                    target = CLUSTERS["flightservice"]
                if target is not None:
                    prev = gt[c]
                    gt[c] = target
                    forced_entity_mapping.append(c)

    # Keep infrastructure / tooling as -1
    for c in class_order:
        if c.startswith("com.acmeair.reporter") or c.startswith("com.acmeair.config"):
            gt[c] = -1
        if c.startswith("com.acmeair.web.dto"):
            gt[c] = -1
        if c.startswith("com.acmeair.morphia.") and ("Converter" in c or c.endswith("DatastoreFactory")):
            gt[c] = -1
        if c in {
            "com.acmeair.service.KeyGenerator",
            "com.acmeair.service.ServiceLocator",
            "com.acmeair.web.AcmeAirApp",
            "com.acmeair.web.AppConfig",
            "com.acmeair.loader.Loader",
            "com.acmeair.config.LoaderREST",
        }:
            gt[c] = -1

    meta = {
        "system": system,
        "authority": {
            "repo": "data/raw/acmeair-quarkus-main",
            "services": [
                "acmeair-authservice-quarkus",
                "acmeair-bookingservice-quarkus",
                "acmeair-customerservice-quarkus",
                "acmeair-flightservice-quarkus",
                "acmeair-mainservice-quarkus",
            ],
            "note": "Monolith classes mapped to IBM microservice boundaries by service interfaces/entities and impl propagation.",
        },
        "cluster_legend": {
            str(CLUSTERS["customerservice"]): "customerservice",
            str(CLUSTERS["bookingservice"]): "bookingservice",
            str(CLUSTERS["flightservice"]): "flightservice",
            str(CLUSTERS["authservice"]): "authservice",
            "-1": "infra/tooling/dto",
        },
        "rules_applied": {
            "seed_assignments": applied,
            "interface_impl_propagation": interface_impl_propagation,
            "entity_impl_forced_by_semantic_name": forced_entity_mapping,
        },
        "summary": _summarize_gt(gt, class_order=class_order),
    }

    return GTResult(gt=gt, meta=meta)


def rebuild_jpetstore() -> GTResult:
    system = "jpetstore"
    class_order = _load_class_order(system)
    # Define clusters: 0 Catalog, 1 Order, 2 Cart, 3 web/gateway/other, -1 infra
    CLUSTERS = {"catalog": 0, "order": 1, "cart": 2, "web": 3}

    gt: Dict[str, int] = {}
    forced_mappings: List[Dict[str, Any]] = []

    # simple keyword rules for controller redistribution
    for c in class_order:
        lower = c.lower()
        mapped = None
        rule = ""
        # treat typical controller / servlet / action classes
        if any(k in lower for k in ("catalog", "product", "category", "item")):
            mapped = CLUSTERS["catalog"]
            rule = "keyword:catalog/product/category/item->catalog"
        elif any(k in lower for k in ("order", "purchase", "checkout")):
            mapped = CLUSTERS["order"]
            rule = "keyword:order/purchase/checkout->order"
        elif any(k in lower for k in ("cart", "shoppingcart")):
            mapped = CLUSTERS["cart"]
            rule = "keyword:cart/shoppingcart->cart"
        # entities / model classes: map to domain based on name hints
        elif ".model." in lower or ".domain." in lower or lower.endswith("entity") or ".entity" in lower:
            # fallback: try to infer
            if "catalog" in lower or "product" in lower:
                mapped = CLUSTERS["catalog"]
                rule = "entity-hint->catalog"
            elif "order" in lower or "purchase" in lower:
                mapped = CLUSTERS["order"]
                rule = "entity-hint->order"
            elif "cart" in lower:
                mapped = CLUSTERS["cart"]
                rule = "entity-hint->cart"
            else:
                mapped = CLUSTERS["web"]
                rule = "entity:unknown->web"
        # default: keep as web/gateway interface
        else:
            mapped = CLUSTERS["web"]
            rule = "fallback:web/gateway"

        gt[c] = int(mapped)
        if rule and not rule.startswith("fallback"):
            forced_mappings.append({"class": c, "to": mapped, "rule": rule})

    meta = {
        "system": system,
        "authority": {
            "repo": "data/raw/jPetStore",
            "note": "Redistribute large controller cluster into Catalog/Order/Cart using lightweight keyword heuristics. Other classes default to web/gateway.",
        },
        "cluster_legend": {
            str(CLUSTERS["catalog"]): "catalog",
            str(CLUSTERS["order"]): "order",
            str(CLUSTERS["cart"]): "cart",
            str(CLUSTERS["web"]): "web/gateway",
            "-1": "infra/unmatched",
        },
        "rules_applied": {
            "forced_keyword_mappings": forced_mappings,
            "notes": [
                "Controller classes matched by keywords (catalog/order/cart) are steered into domain clusters.",
                "Fallback controller/classes remain in web/gateway to avoid over-splitting.",
            ],
        },
        "summary": _summarize_gt(gt, class_order=class_order),
    }

    return GTResult(gt=gt, meta=meta)


def rebuild_daytrader() -> GTResult:
    system = "daytrader"
    class_order = _load_class_order(system)

    # 0 accounts, 1 quotes, 2 portfolios, 3 gateway (business entry), 4 web (UI)
    CLUSTERS = {
        "accounts": 0,
        "quotes": 1,
        "portfolios": 2,
        "gateway": 3,
        "web": 4,
    }

    forced_semantic: List[Dict[str, Any]] = []
    forced_coverage: List[Dict[str, Any]] = []

    def assign_by_rule(c: str) -> Tuple[int, str] | Tuple[None, str]:
        # --- Hard exclude: diagnostics / non-business artifacts ---
        if c.startswith("com.ibm.websphere.samples.daytrader.web.prims."):
            return -1, "infra:web.prims"
        if c.startswith("com.ibm.websphere.samples.daytrader.web.websocket."):
            return -1, "infra:web.websocket"
        # test/build remain infra
        if c.endswith("TestServlet") or c.endswith("TradeBuildDB"):
            return -1, "infra:test/build"

        # config/filter/listener are cross-cutting within the web-service; keep them labeled as web
        if c.endswith("Filter") or c.endswith("Listener") or c.endswith("ContextListener"):
            return CLUSTERS["web"], "web:filter/listener"
        if c.endswith("ConfigServlet") or c.endswith("TradeConfigServlet") or c.endswith("TradeConfigJSF"):
            return CLUSTERS["web"], "web:config"

        # --- Entities: strongest signal, must be fully covered ---
        if ".entities." in c:
            simple = c.rsplit(".", 1)[-1]
            if simple.startswith("Account"):
                return CLUSTERS["accounts"], "entity-prefix:Account*"
            if simple.startswith("Quote"):
                return CLUSTERS["quotes"], "entity-prefix:Quote*"
            if simple.startswith("Holding") or simple.startswith("Order"):
                return CLUSTERS["portfolios"], "entity-prefix:Holding*/Order*"
            return CLUSTERS["portfolios"], "entity:default->portfolios"

        # --- Core service entrypoints (EJB/direct) must be covered ---
        if c.startswith("com.ibm.websphere.samples.daytrader.ejb3.") or c.startswith("com.ibm.websphere.samples.daytrader.direct."):
            return CLUSTERS["gateway"], "layer:ejb3/direct->gateway"

        # --- Root/core classes ---
        if c in {
            "com.ibm.websphere.samples.daytrader.TradeServices",
            "com.ibm.websphere.samples.daytrader.TradeAction",
        }:
            return CLUSTERS["gateway"], "core:TradeServices/TradeAction->gateway"

        # --- Beans: basic domain assignment ---
        if c == "com.ibm.websphere.samples.daytrader.beans.MarketSummaryDataBean" or c.startswith("com.ibm.websphere.samples.daytrader.ejb3.MarketSummary"):
            return CLUSTERS["quotes"], "semantic:MarketSummary->quotes"
        if c == "com.ibm.websphere.samples.daytrader.beans.RunStatsDataBean":
            return CLUSTERS["gateway"], "semantic:RunStats->gateway"

        # --- Web layer: assign to web vs gateway by functional keywords ---
        if c.startswith("com.ibm.websphere.samples.daytrader.web.jsf."):
            # entity/view hints
            s = c.lower()
            if "portfolio" in s or "order" in s or "holding" in s:
                return CLUSTERS["portfolios"], "web.jsf:portfolio/order/holding->portfolios"
            if "market" in s or "summary" in s or "quote" in s:
                return CLUSTERS["quotes"], "web.jsf:quotes/market->quotes"
            if "account" in s or "login" in s:
                return CLUSTERS["accounts"], "web.jsf:account/login->accounts"
            return CLUSTERS["web"], "layer:web.jsf->web"

        if c.startswith("com.ibm.websphere.samples.daytrader.web."):
            # business entrypoints map to gateway
            if c.endswith("TradeAppServlet") or c.endswith("TradeScenarioServlet") or c.endswith("TradeServletAction"):
                return CLUSTERS["gateway"], "semantic:web entry->gateway"
            # other web.* classes: treat as web-service facade
            return CLUSTERS["web"], "layer:web.*->web"

        # Fallback: if under daytrader base package, keep as gateway (better coverage than -1)
        if c.startswith("com.ibm.websphere.samples.daytrader."):
            return CLUSTERS["gateway"], "fallback:daytrader.*->gateway"

        return None, "unmatched"

    gt: Dict[str, int] = {}
    for c in class_order:
        k, rule = assign_by_rule(c)
        if k is None:
            gt[c] = -1
        else:
            gt[c] = k
            if rule.startswith("semantic:") or rule.startswith("entity-prefix") or rule.startswith("web.jsf"):
                forced_semantic.append({"class": c, "to": k, "rule": rule})

    # Enforce 100% coverage for entities/ejb3/direct/core: any -1 in these areas should be assigned
    for c in class_order:
        # entities
        if ".entities." in c and gt.get(c, -1) == -1:
            simple = c.rsplit(".", 1)[-1]
            if simple.startswith("Account"):
                tgt = CLUSTERS["accounts"]
            elif simple.startswith("Quote"):
                tgt = CLUSTERS["quotes"]
            elif simple.startswith("Holding") or simple.startswith("Order"):
                tgt = CLUSTERS["portfolios"]
            else:
                tgt = CLUSTERS["portfolios"]
            prev = gt.get(c, -1)
            gt[c] = tgt
            forced_coverage.append({"class": c, "from": prev, "to": tgt, "rule": "force:entities->cover"})

        # ejb3/direct layers
        if (c.startswith("com.ibm.websphere.samples.daytrader.ejb3.") or c.startswith("com.ibm.websphere.samples.daytrader.direct.")) and gt.get(c, -1) == -1:
            prev = gt.get(c, -1)
            gt[c] = CLUSTERS["gateway"]
            forced_coverage.append({"class": c, "from": prev, "to": CLUSTERS["gateway"], "rule": "force:ejb3/direct->gateway"})

        # core named classes
        if c in {"com.ibm.websphere.samples.daytrader.TradeServices", "com.ibm.websphere.samples.daytrader.TradeAction"} and gt.get(c, -1) == -1:
            prev = gt.get(c, -1)
            gt[c] = CLUSTERS["gateway"]
            forced_coverage.append({"class": c, "from": prev, "to": CLUSTERS["gateway"], "rule": "force:core->gateway"})

    meta = {
        "system": system,
        "authority": {
            "repo": "data/raw/sample.daytrader.microservices-main",
            "workloads": ["accounts", "quotes", "portfolios", "gateway", "web"],
            "note": "High-coverage GT aligned to IBM DayTrader microservices using entity-centered mapping; prims/websocket/test/build excluded as infra (-1). Entities/ejb3/direct/core forced to be covered.",
        },
        "cluster_legend": {
            str(CLUSTERS["accounts"]): "accounts",
            str(CLUSTERS["quotes"]): "quotes",
            str(CLUSTERS["portfolios"]): "portfolios",
            str(CLUSTERS["gateway"]): "gateway",
            str(CLUSTERS["web"]): "web",
            "-1": "infra/unmatched",
        },
        "rules_applied": {
            "semantic_or_entity_forced": forced_semantic,
            "forced_coverage_assignments": forced_coverage,
            "notes": [
                "Entities and ejb3/direct/core classes are force-covered to avoid evidence loss (per user directive).",
            ],
        },
        "summary": _summarize_gt(gt, class_order=class_order),
    }

    return GTResult(gt=gt, meta=meta)


def rebuild_plants() -> GTResult:
    system = "plants"
    class_order = _load_class_order(system)

    # PBW (Plants by WebSphere): 0 customer, 1 catalog, 2 order/cart, 3 web/ui
    CLUSTERS = {
        "customer": 0,
        "catalog": 1,
        "order": 2,
        "web": 3,
    }

    forced: List[Dict[str, Any]] = []

    def assign_by_rule(c: str) -> Tuple[int, str]:
        # Fast path: keep historical package split (bean/jpa/war)
        if c.startswith("com.ibm.websphere.samples.pbw.bean."):
            simple = c.rsplit(".", 1)[-1].lower()
            if "customer" in simple:
                return CLUSTERS["customer"], "pbw.bean:customer"
            if "catalog" in simple or "supplier" in simple:
                return CLUSTERS["catalog"], "pbw.bean:catalog/supplier"
            if "cart" in simple or "order" in simple or "backorder" in simple:
                return CLUSTERS["order"], "pbw.bean:order/cart"
            return CLUSTERS["web"], "pbw.bean:fallback->web"

        if c.startswith("com.ibm.websphere.samples.pbw.jpa."):
            simple = c.rsplit(".", 1)[-1].lower()
            if "customer" in simple:
                return CLUSTERS["customer"], "pbw.jpa:customer"
            if any(k in simple for k in ("inventory", "supplier")):
                return CLUSTERS["catalog"], "pbw.jpa:catalog"
            if any(k in simple for k in ("order", "backorder", "orderitem", "orderkey")):
                return CLUSTERS["order"], "pbw.jpa:order"
            return CLUSTERS["web"], "pbw.jpa:fallback->web"

        if c.startswith("com.ibm.websphere.samples.pbw.war."):
            return CLUSTERS["web"], "pbw.war:web/ui"

        # Fallback keyword mapping for any unexpected class
        lower = c.lower()
        if "customer" in lower:
            return CLUSTERS["customer"], "keyword:customer"
        if any(k in lower for k in ("catalog", "inventory", "supplier")):
            return CLUSTERS["catalog"], "keyword:catalog"
        if any(k in lower for k in ("order", "cart", "checkout", "backorder")):
            return CLUSTERS["order"], "keyword:order/cart"
        return CLUSTERS["web"], "fallback:web"

    gt: Dict[str, int] = {}
    for c in class_order:
        k, rule = assign_by_rule(c)
        gt[c] = int(k)
        # record non-trivial mappings for audit
        if not rule.endswith("fallback->web") and not rule.startswith("fallback"):
            forced.append({"class": c, "to": k, "rule": rule})

    meta = {
        "system": system,
        "authority": {
            "repo": "data/raw/plantsbywebsphere",
            "note": "PlantsByWebSphere (PBW) ground truth rebuilt using package-layer rules (bean/jpa/war) and bounded keyword fallbacks.",
        },
        "cluster_legend": {
            str(CLUSTERS["customer"]): "customer",
            str(CLUSTERS["catalog"]): "catalog",
            str(CLUSTERS["order"]): "order/cart",
            str(CLUSTERS["web"]): "web/ui",
            "-1": "infra/unmatched",
        },
        "rules_applied": {
            "forced_mappings": forced,
            "notes": [
                "Primary mapping uses PBW package layers: pbw.bean/pbw.jpa/pbw.war.",
                "Fallback keyword rules are bounded and recorded for audit.",
            ],
        },
        "summary": _summarize_gt(gt, class_order=class_order),
    }

    return GTResult(gt=gt, meta=meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["acmeair", "daytrader", "jpetstore", "plants"], help="System to rebuild")
    ap.add_argument("--batch", action="store_true", help="Rebuild all supported systems")
    args = ap.parse_args()

    targets = [args.system] if args.system else []
    if args.batch:
        targets = ["acmeair", "daytrader", "jpetstore", "plants"]
    if not targets:
        raise SystemExit("Provide --system or --batch")

    for sys in targets:
        if sys == "acmeair":
            res = rebuild_acmeair()
        elif sys == "daytrader":
            res = rebuild_daytrader()
        elif sys == "jpetstore":
            res = rebuild_jpetstore()
        elif sys == "plants":
            res = rebuild_plants()
        else:
            raise SystemExit(f"Unsupported: {sys}")

        gt_path = ROOT / "data" / "processed" / "groundtruth" / f"{sys}_ground_truth.json"
        meta_path = ROOT / "data" / "processed" / "groundtruth" / f"{sys}_ground_truth_meta.json"
        _write_json(gt_path, res.gt)
        _write_json(meta_path, res.meta)
        print(f"[OK] wrote {gt_path}")
        print(f"[OK] wrote {meta_path}")
        print("[SUMMARY]", json.dumps(res.meta["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
