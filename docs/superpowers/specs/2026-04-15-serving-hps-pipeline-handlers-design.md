# Design: Unified `pipeline_handlers` for Basic Serving and HPS

**Status:** Draft for review
**Date:** 2026-04-15
**Scope:** PaddleX `paddlex/inference/serving/` (basic serving) and `deploy/hps/` (high-stability Triton deployment), sharing logic rooted in the same repository.

## 1. Goals, scope, constraints, non-goals

### 1.1 Goals

1. **Single source of truth** for business logic from parsed inputs to `InferResult` (including markdown, `exports`, visualization images where applicable). HTTP (FastAPI) and HPS (Triton) differ only in transport, concurrency shell, and error surface.
2. **Synchronous core** — shared logic exposed primarily as plain `def` functions for testability; `async` remains only in the HTTP path (`aiohttp`, `PipelineWrapper.infer`, `call_async` as needed).
3. **Full coverage (option B)** — every pipeline that has **both** a `basic_serving` app and an HPS `model.py` must be migrated to the new structure before the initiative is considered complete. Pipelines that exist on **only one side** are listed in the inventory appendix and are **not** required to gain a new deployment path.

### 1.2 Scope (in-repo)

- Add or consolidate modules under `paddlex/inference/serving/`, primarily **`pipeline_handlers/`** (name may be adjusted but the role is fixed: per-pipeline or shared sync orchestration).
- **`paddlex_hps_server`** continues to consume the same source tree as shipped in the HPS image (no separate PyPI package); it imports shared code from `paddlex.inference.serving` rather than duplicating business logic.

### 1.3 Constraints

- **External contracts** — existing JSON shapes, Pydantic schemas, and Triton tensor conventions remain stable; internal refactors must not break clients.
- **Concurrency** — preserve the semantics of `PipelineWrapper` (single worker thread for Paddle inference) and existing HPS batching / grouping / `ThreadPoolExecutor` unless tests prove an equivalent change is safe.
- **HPS packaging** — `paddlex-hps-server` stays image-only; version alignment is ensured by building from the **same repo revision** as `paddlex`, not by independent semver.

### 1.4 Non-goals (this initiative)

- Replacing Triton with HTTP or vice versa.
- Redesigning `paddlex_hps_client` wire protocol.
- Forcing all pipelines into one cross-family mega-handler when behavior genuinely diverges.

---

## 2. Architecture: layers, modules, adapters

### 2.1 Layering (bottom to top)

1. **`schemas/` & `infra/`** — Request/response Pydantic models, storage abstractions, `AIStudio*` envelopes, `generate_log_id`, `call_async`, config. Remain **cross-cutting** and **must not** absorb per-pipeline loops (avoid further “business creep”).
2. **`pipeline_handlers/`** (new, sibling to `basic_serving/`) — **Sync** orchestration: from validated request fields (or pre-parsed `images`, `data_info`, flags) and `predict` outputs to **`InferResult`** (or building blocks thereof). No FastAPI / Triton imports. May call `app_common`-style helpers during migration; see **§2.4**.
3. **`basic_serving/_pipeline_apps/*.py`** — Thin: routes, `PipelineWrapper`, async input fetch (`get_images`), `await pipeline.infer(...)`, `await call_async(handler, ...)` for CPU-heavy shared work, `HTTPException` for client errors.
4. **`deploy/hps/.../model.py`** — Thin: Triton lifecycle, `run` / `run_batch`, grouping, preprocess/postprocess, `protocol.create_aistudio_output_*`; **same** sync handlers as HTTP where business matches.

**Naming:** Prefer aligning module names with schema packages (e.g. `pipeline_handlers/ocr.py` ↔ `schemas/ocr.py`) for discoverability.

### 2.2 Optional: custom exceptions

Introduce a **small** hierarchy under `paddlex.inference.serving` (e.g. `infra/exceptions.py` or `exceptions.py`):

| Type | Role | Adapter behavior |
|------|------|-------------------|
| `ServingValidationError` | Input / business validation → **422** | HTTP: `HTTPException(422, ...)`; HPS: `protocol.create_aistudio_output_without_result(422, ...)` |
| Further subtypes | Only if a distinct **adaptor** branch is needed | Same pattern |

Unhandled exceptions remain **500** with existing logging. `pipeline_handlers` should not catch broadly to hide bugs.

### 2.3 Lifecycle of `app_shared/`

- **Target:** Move document export, markdown refill, and related orchestration into **`pipeline_handlers/`** (e.g. `pipeline_handlers/document_export.py` or `pipeline_handlers/exports/`).
- **Transition:** Optional thin **re-exports** from `app_shared/` to avoid a flag-day rename; follow-up PRs remove `app_shared` once call sites point at `pipeline_handlers`.
- **End state:** **`app_shared` is not a required long-term layer**; eliminate or keep only trivial shims for one release if needed.

### 2.4 Changes to `infra/` and `schemas/`

- **Default:** **No structural change** to schemas; handlers accept existing Pydantic types.
- **`infra`:** Host shared **exceptions** if chosen; keep storage/config/response shells as today.
- **Optional later:** Structured error payload models in `schemas/` only if clients need typed error bodies — **out of scope** for mandatory B delivery unless product requests it.

---

## 3. Inventory, migration waves, PR strategy

### 3.1 Phase 0 — Inventory (blocking)

Author a table **in this spec or a linked checklist** with columns:

- `basic_serving` module
- HPS `model.py` path(s) (multi-stage pipelines: one row per stage)
- Target `pipeline_handlers` module(s)
- Migrated? (checkbox)

No code migration starts until the inventory is reviewed.

### 3.2 Suggested wave order

1. Document-heavy pipelines with high duplication: **OCR, PP-StructureV3, layout_parsing, PaddleOCR-VL**, etc.
2. Medium complexity: **table/seal/doc_preprocessor/PP-DocTranslation**, etc.
3. Detection / segmentation / video / time-series families.
4. Multi-model pipelines (**ChatOCR**, **PP-ShiTuV2**, etc.) — explicit per-stage rows in the inventory.

### 3.3 PR strategy

- **Multiple PRs** are expected; **full coverage (B)** means every inventory row is checked **before** closing the initiative.
- Each PR should be reviewable (subset of pipelines + tests) and revert-friendly.

---

## 4. Testing, regression, and definition of done

### 4.1 Unit tests

- **Sync handler tests** — Given fixed **request-like structs** and **mocked or minimal predict outputs** (fixtures), assert **`InferResult`** (or key fields) match golden JSON/dicts. Prefer tests alongside `pipeline_handlers` or under `tests/`.
- **Exception mapping** — Unit-test that `ServingValidationError` (or chosen type) is translated correctly in a **small** HTTP/Triton adapter test double **if** the mapping logic is non-trivial.

### 4.2 Integration / regression

- Where CI already runs serving or HPS smoke tests, **extend** with migrated pipelines.
- Prefer **before/after** parity checks for at least one request per migrated pipeline (golden response hash or field-wise diff excluding `logId` / timestamps).

### 4.3 Definition of done (initiative)

1. Inventory: all paired rows **migrated** and checked.
2. No duplicate business orchestration left between `basic_serving` and HPS for those rows (except explicitly documented thin glue).
3. `app_shared` removed or only transitional shims with a removal issue filed.
4. Documentation: short developer note (where to add a new pipeline handler; HTTP vs HPS glue). Prefer a small `README.md` under `paddlex/inference/serving/` unless docs team prefers a page under `docs/`.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Subtle behavior drift between HTTP and HPS after refactor | Parity tests; per-PR focused review; shared kwargs builder functions |
| Large PRs | Strict inventory + wave-based merges |
| Exception type proliferation | Start with one validation base type; add subclasses only when adaptors need distinct branches |

---

## 6. Open items (resolved at implementation time)

- Package path for exception classes: default **`paddlex/inference/serving/infra/exceptions.py`** unless a circular-import issue forces a top-level `exceptions.py`.
- Whether to keep temporary `app_shared` re-exports for one release (team preference; default **yes for one release** if many external forks import `app_shared`).

---

## 7. Approval

- [ ] Design reviewed by maintainer(s)
- [ ] Inventory Phase 0 completed before first migration PR

After approval, use **implementation planning** (separate plan) to sequence PRs and assign owners.
