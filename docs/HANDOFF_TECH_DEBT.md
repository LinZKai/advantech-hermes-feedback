# Handoff Tech Debt

交接時**刻意沒有處理**的項目。都不影響目前功能與 tests，但未來值得改善。
每項附「現況 / 為什麼值得改 / 為什麼現在不改 / 未來建議」。

---

## 1. Domain 模組 docstring 大量使用過時的 "a future Reflector step" 措辭

**現況**
`reflector_proposals.py`、`proposal_matching.py`、`case_reflection_input.py`、
`reflector_prompt_context.py` 都是在 `reflector_analyzer.py` 存在**之前**寫的純
contract 模組，docstring 裡有數十處說「the semantic judgment is a future Reflector
step's job」「NOT implemented here」。實際上那個 "future Reflector step" 就是現在的
`reflector_analyzer.py`，已實作。

**為什麼值得改**
接手者讀這些 docstring 會誤以為 Reflector 尚未完成，或搞不清楚時序。

**為什麼現在不改**
這些句子描述的**架構邊界**其實仍正確（該模組本身確實不做語意判斷），錯的只是
時態。全面改寫會產生跨 4 個檔、數十行的純文件 diff，churn 大、易引入新錯、對行為
零幫助——正是「看到可以更漂亮就改」該避免的。

**未來建議**
下次有實質理由碰這些檔時，順手把 "a future Reflector step" 改成 "the Reflector
analyzer（`tools.reflector_analyzer`）"，把 "NOT implemented here" 保留（那是對的）。

---

## 2. `FeedbackStoreV2` 是單一 ~2100 行、涵蓋 10 張表的 god-module

**現況**
`feedback_store_v2.py` 同時是：10 張 v2 table 的全部 CRUD、4 套 CHECK enum 的
taxonomy authority、schema DDL 來源。

**為什麼值得改**
單檔過大，新人不易一眼看出「哪些 method 屬於哪個 domain」；修改任一 domain 都要
在 2000 行裡定位。

**為什麼現在不改**
tests（`test_feedback_store_v2.py` 等多支）與所有 consumer 都 import 這個路徑；
拆檔是高風險大搬移，違反交接原則。taxonomy authority 若拆開更容易 drift。

**未來建議**
若要拆，先抽 `feedback_store_v2/taxonomies.py`（純常數，最安全），再按 domain
（feedback / case / reflector / curator）拆 mixin 或子模組，保持
`FeedbackStoreV2` 對外介面不變。

---

## 3. `retrieval_runtime.py` 同時是 gateway glue 與 Case Routing policy

**現況**
module docstring 只說它是「Phase 3A：把 retrieval_observer 的 parser 輸出接進
FeedbackStoreV2」，但它其實還承載了整個 Case Routing 判定政策
（`_resolve_case_assignment` 的 9 種 `case_assignment_method` 分類、deterministic
case_id 產生、candidate 載入）。

**為什麼值得改**
命名與 docstring 沒反映「Case assignment policy 也住在這裡」，接手者找 routing
邏輯時不會第一個想到這支檔。

**為什麼現在不改**
純命名 / 拆檔問題，改動會牽動 gateway import 點與多支 tests，無行為收益。

**未來建議**
把 Case-assignment policy（`_resolve_case_assignment` 及相關 `CASE_ASSIGNMENT_METHOD_*`
常數）抽成 `case_assignment.py`，`retrieval_runtime.py` 專心做 persistence glue；
並更新 module docstring。

---

## 4. 3 個 vendored full-file Core overlay（`run.py` 約 20000 行）

**現況**
`gateway/run.py`、`gateway/platforms/base.py`、`plugins/platforms/telegram/adapter.py`
是整份 upstream copy，只為插入少數 hook。綁死 Hermes v0.18.0。

**為什麼值得改**
升級 Hermes 要對 3 個大檔重新 diff + 重新套用整合點，成本高且易漏。

**為什麼現在不改**
當時 Hermes v0.18.0 沒有足夠的外部 lifecycle hook / plugin extension point；
在交接前重寫整合方式風險過高。

**未來建議**
- 追蹤新版 Hermes 是否提供正式的 turn-lifecycle hook / plugin extension point；
  有的話把 `run.py` / `base.py` 的整合點改成 plugin，移除 vendored copy。
- Telegram feedback UI 若改由外部 middleware / bot 處理，`adapter.py` overlay 也可移除。
- 短期：README §4 已列出所有整合點的 grep 關鍵字，升級時據此逐一比對。

---

## 5. `migrations/*.sql` 已與 runtime DDL drift，且沒有 runner

**現況**
`001`～`007` 存在但 runtime 不執行；`002` 的 `cases.title` / `cases.product_model`、
`004` 的舊 6 值 diagnosis、`003`/`005` 提到的 `_upgrade_*()` 方法都已不符現狀
（詳見 `DATABASE_DICTIONARY.md` 的〈Migrations vs Runtime DDL〉）。

**為什麼值得改**
兩份 schema 來源並存且不一致，維護者可能誤讀。

**為什麼現在不改**
POC 沒有 persistent DB，schema 靠 `CREATE TABLE IF NOT EXISTS` 重建即可；動
migrations 歷史內容不在交接範圍。

**未來建議**
二擇一：(a) 正式導入 migration runner（依序套用、記錄 schema version），把
`migrations/` 變成真的 source of truth；或 (b) 把 `migrations/` 移到
`docs/schema-history/` 並在每個檔頭加 `ARCHIVED` 標記，明確宣告 Python DDL 為唯一
runtime 來源。

---

## 6. `scripts/reflector_reasoning_smoke_test.py` 是需要 Hermes sandbox 才能跑的手動實驗

**現況**
不在 `overlay/`、不在 Dockerfile、不在 test suite；只有在 built Hermes sandbox 內
`--dry-run` 以外的模式才有意義。

**為什麼值得改**
放在 `scripts/` 根目錄，容易讓人以為是常規工具。

**為什麼現在不改**
它是低風險、對「調 Reflector prompt」有幫助的手動 aid，刪掉略可惜。

**未來建議**
移到 `scripts/dev/` 或類似明確標示「dev-only、需 sandbox」的位置；或在確認不再需要
調整 Reflector prompt 後刪除。
