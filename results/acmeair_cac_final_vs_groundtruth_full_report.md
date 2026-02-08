# AcmeAir：cac-final 分区 vs Ground Truth（全量对比报告）

- Pred 文件：`d:/multimodal_microservice_extraction/data/processed/fusion/acmeair_cac-final_partition.json`
- GT 文件：`d:/multimodal_microservice_extraction/data/processed/groundtruth/acmeair_ground_truth.json`

## 0. 关键说明
- GT 中 `-1` 表示「不纳入聚类/不参与评估的类」（如 config/dto/reporter/工具类等）。
- Pred（cac-final）会给所有类分配一个簇 id。
- 主要统计采用：仅评估 **GT>=0 且 Pred 也存在** 的类（最常见也最公平）。

## 1. 覆盖情况（key 集合差异）
- GT 类数：**71**
- Pred 类数：**71**
- 并集：**71**
- GT 有但 Pred 缺失：**0**
- Pred 有但 GT 缺失：**0**

## 2. 整体统计（仅 GT>=0 且 Pred 存在）
- 参与评估类数：**41**
- 不一致（GT!=Pred）：**36**
- 逐类一致率（Accuracy，仅作直观参考）：**0.122**

## 3. 混淆矩阵（GT 行，Pred 列）
| GT\Pred | 0 | 1 | 2 | 3 | 4 | RowSum |
|---|---|---|---|---|---|---|
| 0 | 3 | 1 | 4 | 0 | 3 | 11 |
| 1 | 3 | 1 | 4 | 0 | 2 | 10 |
| 2 | 5 | 1 | 1 | 4 | 4 | 15 |
| 3 | 1 | 2 | 1 | 0 | 1 | 5 |

## 4. Pred 簇纯度（多数 GT 标签占比）
| Pred簇 | 簇大小 | 多数GT | 多数计数 | 纯度 |
|---|---|---|---|---|
| 0 | 12 | 2 | 5 | 0.417 |
| 1 | 5 | 3 | 2 | 0.400 |
| 2 | 10 | 1 | 4 | 0.400 |
| 3 | 4 | 2 | 4 | 1.000 |
| 4 | 10 | 2 | 4 | 0.400 |

## 5. GT=-1 但 Pred 强行分簇
- GT=-1 且 Pred 存在：**30**
强行分簇最多的包（前20）：
- com.acmeair.web.dto: 10
- com.acmeair.reporter.parser: 5
- com.acmeair.reporter.parser.component: 4
- com.acmeair.morphia: 3
- com.acmeair.config: 2
- com.acmeair.service: 2
- com.acmeair.web: 2
- com.acmeair.loader: 1
- com.acmeair.reporter: 1

## 6. 不一致 Top（按包统计）
不一致最多的包（前30）：
- com.acmeair.entities: 7
- com.acmeair.morphia.entities: 7
- com.acmeair.wxs.entities: 7
- com.acmeair.service: 4
- com.acmeair.web: 4
- com.acmeair.morphia.services: 3
- com.acmeair.loader: 2
- com.acmeair.wxs.service: 2

## 7. 不一致明细（前300条）
| Class | GT | Pred |
|---|---|---|
| com.acmeair.entities.Booking | 1 | 2 |
| com.acmeair.entities.BookingPK | 1 | 2 |
| com.acmeair.entities.Customer | 0 | 2 |
| com.acmeair.entities.CustomerAddress | 0 | 2 |
| com.acmeair.entities.CustomerSession | 3 | 2 |
| com.acmeair.entities.Flight | 2 | 3 |
| com.acmeair.entities.FlightSegment | 2 | 3 |
| com.acmeair.loader.CustomerLoader | 0 | 2 |
| com.acmeair.loader.FlightLoader | 2 | 3 |
| com.acmeair.morphia.entities.AirportCodeMappingImpl | 2 | 4 |
| com.acmeair.morphia.entities.BookingImpl | 1 | 4 |
| com.acmeair.morphia.entities.CustomerAddressImpl | 0 | 4 |
| com.acmeair.morphia.entities.CustomerImpl | 0 | 4 |
| com.acmeair.morphia.entities.CustomerSessionImpl | 3 | 4 |
| com.acmeair.morphia.entities.FlightImpl | 2 | 4 |
| com.acmeair.morphia.entities.FlightSegmentImpl | 2 | 4 |
| com.acmeair.morphia.services.BookingServiceImpl | 1 | 4 |
| com.acmeair.morphia.services.CustomerServiceImpl | 0 | 4 |
| com.acmeair.morphia.services.FlightServiceImpl | 2 | 4 |
| com.acmeair.service.BookingService | 1 | 2 |
| com.acmeair.service.CustomerService | 0 | 2 |
| com.acmeair.service.FlightService | 2 | 3 |
| com.acmeair.service.TransactionService | 1 | 2 |
| com.acmeair.web.CustomerREST | 0 | 1 |
| com.acmeair.web.FlightsREST | 2 | 1 |
| com.acmeair.web.LoginREST | 3 | 1 |
| com.acmeair.web.RESTCookieSessionFilter | 3 | 1 |
| com.acmeair.wxs.entities.AirportCodeMappingImpl | 2 | 0 |
| com.acmeair.wxs.entities.BookingImpl | 1 | 0 |
| com.acmeair.wxs.entities.BookingPKImpl | 1 | 0 |
| com.acmeair.wxs.entities.CustomerSessionImpl | 3 | 0 |
| com.acmeair.wxs.entities.FlightImpl | 2 | 0 |
| com.acmeair.wxs.entities.FlightPKImpl | 2 | 0 |
| com.acmeair.wxs.entities.FlightSegmentImpl | 2 | 0 |
| com.acmeair.wxs.service.BookingServiceImpl | 1 | 0 |
| com.acmeair.wxs.service.FlightServiceImpl | 2 | 0 |
