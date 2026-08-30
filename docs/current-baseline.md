# Current Feedback Baseline

> **本檔案已由 [`../README.md`](../README.md) 與
> [`DATABASE_DICTIONARY.md`](DATABASE_DICTIONARY.md) 取代。**
> 保留檔名是因為 `overlay/tools/retrieval_runtime.py` 的 docstring 仍引用此路徑。
> 早期的 POC 限制清單（無 negative reason、無 suggestion、無 dashboard API 等）
> **均已不再適用**——請一律以 README 與現行 code 為準。

## 目前實際狀態（摘要）

- Universal Feedback：👍/👎、負向原因（5 個 reason code）、選填文字建議皆已實作。
- 每個 Hermes turn 自動落地 `sessions` / `cases` / `turns` / `retrieval_runs`。
- Case Routing（`case-router-v1`）、Case Analysis、Reflector、Curator、Dashboard API
  皆已實作。
- Case Analysis / Reflector / Curator / Apply 為**手動觸發，無 scheduler**。
- 3 個 Hermes Core 檔案以 full-file overlay 覆蓋（`gateway/run.py`、
  `gateway/platforms/base.py`、`plugins/platforms/telegram/adapter.py`），綁定
  Hermes v0.18.0。

細節請見 README 各章節。
