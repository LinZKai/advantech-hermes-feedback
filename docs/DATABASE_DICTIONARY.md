# Database Dictionary

本文件整理 `advantech-hermes-feedback` 使用的 SQLite Database Table 與主要欄位用途。

Database：

```text
/sandbox/.hermes/data/support_feedback.db
```

目前共有 11 張 Table。

> Runtime Schema 以 `feedback_storage.py` 與 `feedback_store_v2.py` 中的 Python DDL 為準。  
> `migrations/` 為開發過程中的 Schema 紀錄，目前 Runtime 不會執行這些 SQL。

---

# Table Overview

| Table | 用途 |
|---|---|
| `feedback_runs` | Telegram Feedback 互動過程 |
| `sessions` | 一段對話 Session |
| `cases` | 一個獨立技術問題 |
| `turns` | 一次 User Question / Hermes Answer |
| `retrieval_runs` | Foundry IQ Retrieval 執行情況 |
| `feedback` | 整理後的 Feedback 資料 |
| `case_analysis` | Case Analysis 結果 |
| `reflection_runs` | Reflector 每次分析執行紀錄 |
| `improvement_proposals` | 持續追蹤的改善議題 |
| `proposal_observations` | Reflector 每次對 Proposal 的觀察 |
| `curator_changes` | Curator 產生的 Agent 修改建議 |

---

# `feedback_runs`

Telegram Feedback UI 使用的資料表。

主要保存「這次 Feedback 是誰、在哪個 Chat、對哪一則回答，以及使用者最後給了什麼 Feedback」。

| Column | 說明 |
|---|---|
| `run_id` | 一次 Feedback 流程的識別碼 |
| `chat_id` | Telegram Chat ID |
| `resolved` | 舊版 Feedback 流程使用的欄位 |
| `created_at` | Feedback Prompt 建立時間 |
| `submitted_at` | 使用者送出 Feedback 的時間 |
| `turn_key` | 對應的 Turn |
| `telegram_user_id` | 原始提問者的 Telegram User ID |
| `feedback_message_id` | Feedback Prompt 的 Message ID |
| `helpful` | 是否有幫助 |
| `reason_code` | 負評原因 |
| `suggestion_text` | 使用者補充建議 |
| `feedback_send_status` | Feedback Prompt 是否成功送出 |

> `feedback_runs` 目前仍由 Telegram Feedback 流程直接使用，不是已廢棄 Table。

---

# `sessions`

代表一段平台對話。

一個 Session 中可以包含多個不同的 Cases。

| Column | 說明 |
|---|---|
| `session_id` | Session 唯一識別碼 |
| `platform` | 使用的平台，目前主要為 Telegram |
| `platform_chat_id` | 平台 Chat ID |
| `created_at` | Session 建立時間 |
| `updated_at` | 最近更新時間 |

---

# `cases`

代表一個獨立的技術支援問題。

同一個 Case 可以包含多個 Turns。

| Column | 說明 |
|---|---|
| `case_id` | Case 唯一識別碼 |
| `session_id` | 此 Case 所屬 Session |
| `created_at` | Case 建立時間 |
| `updated_at` | Case 更新時間 |

Case Title、Product Model、Diagnosis 等分析內容不放在這張表，而是存在 `case_analysis`。

---

# `turns`

代表一次完整的使用者提問與 Hermes 回答。

| Column | 說明 |
|---|---|
| `turn_id` | Turn 唯一識別碼 |
| `case_id` | 此 Turn 所屬 Case |
| `session_id` | 此 Turn 所屬 Session |
| `platform_user_id` | 使用者的平台 ID |
| `platform_user_message_id` | 使用者問題的 Message ID |
| `platform_assistant_message_id` | Hermes 回答的 Message ID |
| `question_text` | 使用者問題 |
| `answer_text` | Hermes 最終回答 |
| `feedback_eligible` | 此 Turn 是否需要顯示 Feedback |
| `retrieval_observation_status` | Retrieval 資料是否有被完整記錄 |
| `retrieval_observation_reason` | Retrieval 紀錄不完整時的原因 |
| `support_config_commit` | 當時使用的 Support Config Version |
| `feedback_code_commit` | 當時使用的 Feedback Code Version |
| `hermes_version` | 當時 Hermes Version |
| `model` | 使用的模型 |
| `provider` | 使用的 Model Provider |
| `case_assignment_method` | 此 Turn 如何被分配到 Case |
| `case_assignment_confidence` | Case Routing 信心程度 |
| `case_assignment_classifier_version` | Case Routing Version |
| `case_assignment_overridden_by` | 預留給未來人工修改 Case Assignment |
| `case_assignment_overridden_at` | 預留給未來人工修改 Case Assignment 的時間 |
| `created_at` | Turn 建立時間 |
| `updated_at` | Turn 更新時間 |

---

# `retrieval_runs`

保存一次 Foundry IQ Retrieval 的執行情況。

只保存 Retrieval 狀態與數量，不保存 Knowledge Base 文件內容。

| Column | 說明 |
|---|---|
| `retrieval_id` | Retrieval 唯一識別碼 |
| `turn_id` | 所屬 Turn |
| `invocation_order` | 此 Turn 中第幾次 Retrieval |
| `tool_call_id` | Tool Call ID |
| `request_attempted` | 是否真的送出 Retrieval Request |
| `execution_status` | Retrieval 執行結果 |
| `foundry_iq_ok` | Foundry IQ 是否成功 |
| `observation_status` | Retrieval 紀錄是否完整 |
| `observation_reason` | 紀錄不完整的原因 |
| `error_code` | 發生錯誤時的 Error Code |
| `http_status` | HTTP Status |
| `result_count` | 找到的 Document 數量 |
| `reference_count` | Reference 數量 |
| `foundry_schema_version` | Foundry IQ Result Schema Version |
| `created_at` | Retrieval 紀錄時間 |

---

# `feedback`

將 Telegram 收到的 Feedback 整理成後續分析容易使用的格式。

一個 Turn 最多對應一筆 Feedback。

| Column | 說明 |
|---|---|
| `feedback_id` | Feedback 唯一識別碼 |
| `turn_id` | Feedback 對應的 Turn |
| `helpful` | 是否有幫助 |
| `reason_code` | 負評原因 |
| `suggestion_text` | 使用者補充建議 |
| `feedback_policy_version` | 使用的 Feedback 規則版本 |
| `submitted_at` | Feedback 提交時間 |

### `feedback_runs` 與 `feedback` 的差別

- `feedback_runs`：Telegram Feedback 互動流程使用
- `feedback`：整理後提供 Case Analysis、Reflector 與 Dashboard 使用

目前兩張 Table 都有使用，因此都保留。

---

# `case_analysis`

保存 Case Analysis 的結果。

系統會整理一個 Case 中的 Turns、Retrieval 與 Feedback，再由 LLM 分析問題類型、Diagnosis 與 Product 等資訊。

| Column | 說明 |
|---|---|
| `analysis_id` | 此次 Case Analysis 識別碼 |
| `case_id` | 被分析的 Case |
| `case_title` | Case 標題 |
| `issue_summary` | 問題摘要 |
| `issue_type` | 問題類型 |
| `issue_type_confidence` | Issue Type 信心程度 |
| `diagnosis` | 分析出的問題 Diagnosis |
| `diagnosis_confidence` | Diagnosis 信心程度 |
| `product_model` | 產品型號 |
| `product_source` | Product Model 的來源 |
| `product_confidence` | Product Model 信心程度 |
| `evidence_json` | 分析所依據的 Evidence |
| `analysis_version` | Case Analysis Version |
| `analyzed_at` | 分析時間 |
| `source_evidence_watermark` | 本次分析所看到的最新資料時間 |

同一個 Case 可以被重新分析，因此可能存在多筆 Case Analysis。

---

# `reflection_runs`

保存每次 Reflector 執行的基本資訊。

| Column | 說明 |
|---|---|
| `reflection_run_id` | Reflection Run 唯一識別碼 |
| `window_start` | 本次分析範圍開始時間 |
| `window_end` | 本次分析範圍結束時間 |
| `started_at` | Reflector 開始時間 |
| `completed_at` | Reflector 完成時間 |
| `status` | 執行狀態 |
| `analyzed_case_count` | 本次分析的 Case 數量 |
| `material_change_detected` | 是否發現值得改善的變化 |
| `run_summary` | 本次 Reflector 分析摘要 |
| `reflector_version` | Reflector Version |

---

# `improvement_proposals`

保存 Reflector 找出的改善議題。

一個 Proposal 代表一個持續追蹤的問題。

| Column | 說明 |
|---|---|
| `proposal_id` | Proposal 唯一識別碼 |
| `improvement_target` | 改善方向 |
| `title` | 改善議題標題 |
| `review_status` | 人工審查狀態 |
| `created_at` | Proposal 建立時間 |

`improvement_target` 主要分為：

- `knowledge`
- `agent_behavior`
- `retrieval`
- `workflow`
- `other`

目前只有 `agent_behavior` 類型的 Proposal 會進一步交給 Curator。

---

# `proposal_observations`

保存每次 Reflector 對某個 Proposal 的觀察結果。

簡單來說：

> Proposal 是「正在追蹤什麼問題」  
> Observation 是「這次又觀察到這個問題怎麼樣」

| Column | 說明 |
|---|---|
| `observation_id` | Observation 唯一識別碼 |
| `proposal_id` | 對應的 Proposal |
| `reflection_run_id` | 此 Observation 來自哪次 Reflector |
| `trend` | 問題目前的變化趨勢 |
| `pattern_summary` | 觀察到的重複問題 |
| `possible_cause` | 可能原因 |
| `recommended_improvement` | 建議改善方式 |
| `expected_benefit` | 預期改善效果 |
| `limitations` | 本次分析限制 |
| `supporting_case_ids_json` | 支持此觀察的 Cases |
| `supporting_case_count` | Supporting Cases 數量 |
| `confidence` | 此 Observation 的信心程度 |
| `observed_at` | 觀察時間 |

---

# `curator_changes`

保存 Curator 根據 Improvement Proposal 產生的 Agent 修改建議。

目前修改目標為：

```text
/sandbox/AGENTS.md
```

Curator 產生建議後，仍需要人工 Review 才能 Apply。

| Column | 說明 |
|---|---|
| `change_id` | Curator Change 唯一識別碼 |
| `proposal_id` | 此修改來自哪個 Proposal |
| `target_file` | 要修改的檔案 |
| `change_type` | 修改類型 |
| `rationale` | 為什麼建議這個修改 |
| `before_content` | 修改前的 AGENTS.md |
| `proposed_content` | 建議修改後的 AGENTS.md |
| `expected_effect` | 預期改善效果 |
| `confidence` | Curator 信心程度 |
| `status` | 修改目前的審查 / 套用狀態 |
| `created_at` | Change 建立時間 |
| `reviewed_at` | 人工 Review 時間 |
| `applied_at` | 修改實際套用時間 |

---

# Table Relationship

整體可以簡單理解為：

```text
sessions
   ↓
cases
   ↓
turns
   ├── retrieval_runs
   └── feedback

cases
   ↓
case_analysis
   ↓
Reflector

reflection_runs
   ↓
proposal_observations
   ↑
improvement_proposals
   ↓
curator_changes
```

另外：

```text
feedback_runs
     ↓
   feedback
```

前者負責 Telegram Feedback Interaction，後者負責後續分析。

---

# Schema Note

目前 Runtime Schema 以：

```text
overlay/tools/feedback_storage.py
overlay/tools/feedback_store_v2.py
```

中的 Python DDL 為準。

`migrations/*.sql` 為開發過程中的 Schema 紀錄，目前 Runtime 不會執行這些 Migration。
