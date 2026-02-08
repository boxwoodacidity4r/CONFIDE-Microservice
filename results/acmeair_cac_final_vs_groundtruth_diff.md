# AcmeAir: `cac-final` partition vs Ground Truth (excerpt comparison)

> Based on the provided excerpts (both cover mostly the same class set).  
> `ground_truth.json` uses `-1` to denote “out of scope / not clustered” (utility/config/dto/etc).  
> `cac-final_partition.json` assigns every class to a cluster id `{0..4}`.

## How to read
- **GT** = label in `data/processed/groundtruth/acmeair_ground_truth.json`
- **Pred** = label in `data/processed/fusion/acmeair_cac-final_partition.json`
- **Mismatch** means `GT != Pred` **for GT >= 0** (i.e., for classes that are considered in-scope by GT).
- **GT=-1 but Pred!=?** means GT says “ignore/out-of-scope” but prediction still forced it into a cluster.

---

## 1) In-scope GT classes (GT >= 0) that mismatch in the excerpt

| Class | GT | Pred |
|---|---:|---:|
| `com.acmeair.entities.Booking` | 1 | 2 |
| `com.acmeair.entities.CustomerAddress` | 0 | 2 |
| `com.acmeair.loader.CustomerLoader` | 0 | 2 |
| `com.acmeair.loader.FlightLoader` | 2 | 3 |
| `com.acmeair.morphia.entities.CustomerAddressImpl` | 0 | 4 |
| `com.acmeair.morphia.entities.CustomerImpl` | 0 | 4 |
| `com.acmeair.morphia.entities.CustomerSessionImpl` | 3 | 4 |
| `com.acmeair.morphia.services.BookingServiceImpl` | 1 | 4 |
| `com.acmeair.morphia.services.CustomerServiceImpl` | 0 | 4 |
| `com.acmeair.morphia.services.FlightServiceImpl` | 2 | 4 |
| `com.acmeair.service.BookingService` | 1 | 2 |
| `com.acmeair.service.CustomerService` | 0 | 2 |
| `com.acmeair.service.FlightService` | 2 | 3 |
| `com.acmeair.wxs.service.BookingServiceImpl` | 1 | 0 |
| `com.acmeair.wxs.service.FlightServiceImpl` | 2 | 0 |
| `com.acmeair.web.BookingsREST` | 1 | 0 |
| `com.acmeair.web.CustomerREST` | 0 | 1 |
| `com.acmeair.web.FlightsREST` | 2 | 1 |
| `com.acmeair.web.LoginREST` | 3 | 1 |
| `com.acmeair.web.RESTCookieSessionFilter` | 3 | 1 |

Notes:
- The largest systematic disagreement in the excerpt is that many `morphia.*` impl/service classes that GT distributes across {0,1,2,3} are all mapped to **Pred=4**.
- Several REST controllers (`web.*REST`) are placed into **Pred=0/1**, while GT uses business-aligned labels {0..3} (and also marks many DTOs as -1).

---

## 2) GT out-of-scope classes (GT = -1) that still get clustered in Pred (excerpt)

These are classes GT explicitly excludes, but `cac-final` still assigns to some cluster:

- `com.acmeair.config.AcmeAirConfiguration` (GT=-1, Pred=3)
- `com.acmeair.config.LoaderREST` (GT=-1, Pred=1)
- `com.acmeair.loader.Loader` (GT=-1, Pred=3)
- `com.acmeair.morphia.BigDecimalConverter` (GT=-1, Pred=4)
- `com.acmeair.morphia.BigIntegerConverter` (GT=-1, Pred=4)
- `com.acmeair.morphia.DatastoreFactory` (GT=-1, Pred=4)
- `com.acmeair.reporter.*` and `com.acmeair.reporter.parser.*` (GT=-1, Pred=1/3/4)
- DTOs in `com.acmeair.web.dto.*` (GT=-1, Pred=1/3)
- `com.acmeair.service.KeyGenerator` (GT=-1, Pred=2)
- `com.acmeair.service.ServiceLocator` (GT=-1, Pred=2)
- `com.acmeair.web.AcmeAirApp` (GT=-1, Pred=1)
- `com.acmeair.web.AppConfig` (GT=-1, Pred=1)

This matters because evaluation typically either:
- filters both sides to the same “class universe”, or
- treats GT=-1 as “unassigned”, which can penalize predictions that force assignments.

---

## 3) Where they agree (from the excerpt)

A few in-scope classes do align exactly in the excerpt:
- `com.acmeair.entities.AirportCodeMapping` (GT=2, Pred=2)
- `com.acmeair.entities.Flight` (GT=2, Pred=3) **(actually mismatch)**
- `com.acmeair.entities.FlightSegment` (GT=2, Pred=3) **(mismatch)**
- `com.acmeair.morphia.entities.AirportCodeMappingImpl` (GT=2, Pred=4) **(mismatch)**

In the excerpt, the only clear exact match among those key entities is `AirportCodeMapping`.

---

## 4) Quick interpretation (cluster semantics)

From the excerpt, the predicted clusters look like they are grouping more by **technical layer/implementation**:
- Pred=4: many `morphia.*` persistence/impl classes
- Pred=0: `wxs.*` service/entity impls + some REST (`BookingsREST`)
- Pred=1: web/app/config/dto + auth/session-related (`LoginREST`, `RESTCookieSessionFilter`)
- Pred=2/3: core entities/services/loader/reporter mixed

Whereas GT labels appear more **business-capability aligned** (Customer / Booking / Flight / Session/Login).

---

## If you want an exact full diff (recommended)
I can generate an exact report over the *entire* JSONs (not just the excerpt):
- counts of (GT=-1 forced assignment)
- confusion matrix
- top mismatched packages/classes
- metrics under two evaluation policies:
  1) **filter to GT>=0 only**
  2) **include GT=-1 as “unassigned”**
