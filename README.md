# Local Document RAG System (Nexus)

A Streamlit local-document research chatbot. The system extracts uploaded files,
embeds their chunks into persistent ChromaDB, performs hybrid vector/BM25 retrieval,
coordinates specialized agents with LangGraph, and generates document-grounded
answers without exposing chunk IDs or source metadata in user-facing text.

## Current Architecture

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI<br/>app.py]
    UI --> API[run_rag()<br/>retrieval_agent.py]
    API --> G[LangGraph Orchestrator<br/>agents/graph.py]

    subgraph INGESTION[Document Ingestion and Indexing]
        F[Uploaded PDF / DOCX / CSV / JSON / TXT / MD]
        F --> L[Document Loader<br/>document_loader.py]
        L --> PDF[pypdf<br/>PDF text]
        L --> PT[pdfplumber<br/>PDF tables]
        L --> OCR[PyMuPDF + Pillow + pytesseract<br/>Scanned PDF OCR]
        L --> DOCX[python-docx<br/>DOCX tables]
        L --> CSV[Python csv<br/>CSV tables]
        L --> JSON[Python json<br/>JSON records]
        PDF --> P[ExtractedPage<br/>text + structured_data]
        PT --> P
        OCR --> P
        DOCX --> P
        CSV --> P
        JSON --> P
        P --> C[Chunker<br/>1,000 chars / 150 overlap]
        C --> E[MiniLM Embeddings]
        C --> V[(ChromaDB<br/>Persistent Vector Index)]
        E --> V
    end

    INGESTION -. Indexed local documents .-> G
    G --> O[Orchestrator<br/>Query + History]
    O -->|Direct reply / clarify| DIRECT[Direct response]
    O -->|Complex comparison / multi-part| PLAN[One LLM planning call]
    O -->|Full-document summary| S[Summarizer Agent]
    O -->|Other document questions| EX[Executor Agent]
    PLAN --> EX

    EX --> QT[Question type + follow-up resolution]
    QT --> HS[Hybrid Search]
    HS --> V
    HS --> BM25[BM25<br/>Lexical Search]
    V --> SCORE[Hybrid Score<br/>65% Vector + 35% BM25]
    BM25 --> SCORE
    SCORE --> CUR[Curator<br/>Dedupe / Rerank / Expand]
    CUR --> CE{CrossEncoder<br/>Available?}
    CE -->|Yes| CER[CrossEncoder Reranking]
    CE -->|No| HR[Deterministic Hybrid Reranking]
    CER --> CTX[Final Evidence Context]
    HR --> CTX
    CTX --> W[Writer Agent<br/>Grounded Draft]
    W --> VER[Claim-level Verifier]
    VER -->|Supported| ANS[Grounded Answer<br/>No Source Metadata + Exact Table Values]
    VER -->|Unsupported claims| DROP[Remove unsupported claims]
    DROP --> ANS

    S --> LOAD[Load All Chunks from One Document]
    LOAD --> MAP[Gemini Flash<br/>60-chunk batch summaries]
    MAP --> REDUCE[Gemini Flash<br/>Final synthesis]
    REDUCE --> CACHE[(SQLite<br/>Summary Cache + Run State)]
    CACHE --> SUM[Whole-Document Summary]
    SUM --> VER

    G -.-> TRACE[(retrieval_trace.jsonl)]
    UI -.-> CONV[(conversations.json)]

    classDef ui fill:#2563EB,stroke:#1E3A8A,color:#FFFFFF,stroke-width:2px;
    classDef orchestrator fill:#7C3AED,stroke:#4C1D95,color:#FFFFFF,stroke-width:3px;
    classDef agent fill:#F59E0B,stroke:#92400E,color:#111827,stroke-width:2px;
    classDef retrieval fill:#10B981,stroke:#065F46,color:#FFFFFF,stroke-width:2px;
    classDef tool fill:#06B6D4,stroke:#155E75,color:#FFFFFF,stroke-width:2px;
    classDef storage fill:#64748B,stroke:#1E293B,color:#FFFFFF,stroke-width:2px;
    classDef summary fill:#EC4899,stroke:#831843,color:#FFFFFF,stroke-width:2px;
    classDef output fill:#22C55E,stroke:#166534,color:#FFFFFF,stroke-width:3px;
    classDef decision fill:#F97316,stroke:#9A3412,color:#FFFFFF,stroke-width:2px;

    class U,UI,API ui;
    class G orchestrator;
    class O,PLAN,EX,QT,W,VER agent;
    class CUR,HS,SCORE,HR retrieval;
    class L,PDF,PT,OCR,DOCX,CSV,JSON,P,C,E,BM25,CER tool;
    class V,CACHE,TRACE,CONV storage;
    class S,LOAD,MAP,REDUCE,SUM summary;
    class ANS,OUT,CTX output;
    class CE decision;
```

**Legend:** blue = UI/API, purple = orchestrator, yellow = LLM agents,
green = retrieval, cyan = document/search tools, gray = storage, pink =
summarization, and bright green = final output.


`run_rag(question, history=...)` remains the stable application entry point.
`agents/graph.py` owns orchestration; `agents/retrieval_agent.py` is a
backward-compatible shim. The Orchestrator handles simple questions,
follow-ups, direct replies, clarifications, and full-document summaries without
an LLM planning call. Only complex comparison or multi-part queries require an
LLM planning call. Full-document summaries go to the Summarizer Agent; other
document questions go to the Executor Agent. Set `USE_MULTI_AGENT_RAG = False`
in `config.py` to use the pre-multi-agent retrieval → curate → answer pipeline.

### Implemented summary and state flow

Summary questions now use a dedicated full-document map/reduce path:

```text
Orchestrator
  → pass the current query and conversation history to the Summarizer
  → resolve a document from the query, recent history, or document search
  → ask which document when the target remains ambiguous
  → load every indexed chunk for that document in source order
  → summarize consecutive batches of up to 200 chunks with Gemini Flash Lite
  → verify each MAP summary against only its source batch with Cohere
  → reduce the batch summaries into one final summary
  → SQLite summary cache
  → return the final summary
```

The summary path is intentionally separate from ordinary question retrieval.
The Summarizer receives the current query and conversation history. It first
uses an explicitly named document, then looks for a document named in recent
conversation; a single-document library is selected automatically. If the
request is generic and multiple documents remain plausible, it asks the user
which one to summarize. Otherwise, it uses the topical query to identify one
document. Every indexed chunk in the selected document is passed to Gemini in
source-ordered batches, then the batch summaries are combined in a final
Gemini call. Each request is checked against the model's input-token context
limit; summary requests use a SQLite-backed rolling input
budget of 180,000 TPM, below the observed free-tier quota. When capacity is
exhausted, calls wait instead of dropping document content. Temporary Gemini
429/503 responses retry using the provider's retry delay when available; SDK
internal retries are disabled so every generation attempt is metered. A single
request above the target budget waits for an otherwise-empty window and is
allowed up to the observed 250,000-token provider ceiling.
If a map or reduce request exceeds that ceiling, map batches are split and
reduce synthesis falls back to hierarchical groups rather than failing the
research request.

The final output rails validate citation references against the retrieved
document/chunk/page metadata, check summary section and fact coverage, validate
the answer response shape, and record rail latency and current token usage.
These checks are deterministic and make no extra model calls; the output rail
retry limit is zero. Summary fact coverage is a bounded heuristic report, not a
second semantic-verification pass.
Numbered findings and sentence-like list items are not treated as mandatory
section headings, and a remaining heading-match miss after the map repair pass
is logged without aborting the document summary.
The terminal prints elapsed times and progress for document selection, chunk
loading, cache lookup, each map batch, token counting, TPM pacing, Gemini calls,
and final reduction.
Query and model telemetry is recorded with OpenTelemetry spans and metrics.
The workflow panel uses the root query span for total wall-clock latency and
shows individual (potentially nested) stage spans separately. Model spans
include provider/model, operation, stage, duration, retry counts, and provider
token usage; prompt and source-document contents are not recorded. Gemini token
count requests are traced separately from generation and are not double-counted
as generated usage. OTLP export is disabled by default; set
`OTEL_EXPORTER_OTLP_ENDPOINT` to export traces and metrics to an OpenTelemetry
collector. A bounded local span buffer supports the in-app workflow inspector.
If the final synthesis exceeds the model context, the user is asked before
additional reduction calls are attempted. Each successful map batch is
checkpointed in SQLite, so an interrupted run reuses saved batches and resumes
at the first incomplete batch. Completed summaries are retained permanently
and reused for the same document version and model; changed document content
or structured data produces a new summary. The cache preserves both the
display-safe summary and its raw cited draft. Generated summaries use the same
verifier as ordinary answers. Exact citation-backed extracts are checked
deterministically. Other claims are verified together in one Cohere request
against retrieved passages and previously saved web evidence; one targeted
batch may use additional already-retrieved evidence when fewer than three
claims are supported. Verification does not perform live web search. Keyword
ranking selects evidence locally but is not treated as proof of support.
Generated summaries use five sections: **Overview**, **Main Topics**, **Key
Findings and Contributions**, **Challenges and Limitations**, and **Conclusion**.
LLM architecture, pre-training, adaptation/fine-tuning, utilization, and
evaluation are covered under Main Topics only when the source discusses them.
All claims and any stated limitations must be grounded in the selected document.
Chunk references remain internal metadata and are stripped from the displayed
summary and answer text.
Large source documents can still be summarized in one call because only relevant
chunks are sent. The selected passage payload is preflight-limited to 350,000
characters; if that limit is exceeded, the app explains that multi-call
summarization or a narrower query is needed.

`memory/state_store.py` stores permanent summary results, per-batch resume
checkpoints, and durable per-run status in `data/nexus_state.sqlite3`. ChromaDB
remains the document index, while SQLite is used for request metadata and
reusable summaries.

## Project structure

```text
multi_agent_research/
├── app.py                         # Streamlit user interface
├── config.py                      # Models, limits, paths, and feature flags
├── telemetry.py                   # OpenTelemetry spans, metrics, and local trace capture
├── requirements.txt               # Python dependencies
├── README.md                      # Architecture and operating guide
├── agents/
│   ├── graph.py                   # Main LangGraph orchestrator and shared state
│   ├── executor.py                # Single-executor query workflow
│   ├── router.py                  # Router agent and route classification
│   ├── decomposer.py              # Compound-question decomposition step
│   ├── retriever_worker.py        # Reusable plan/search/grade/retry subgraph
│   ├── curator.py                 # Deterministic dedupe, rerank, and expansion
│   ├── writer.py                  # Grounded answer-generation step
│   ├── summarizer.py              # Gemini full-document map/reduce summarizer
│   ├── verifier.py                # Claim-verification agent
│   ├── schemas.py                 # Pydantic contracts between graph stages
│   └── retrieval_agent.py         # Legacy-compatible run_rag() shim
├── gurdrails/
│   ├── gurdrailsmanager.py        # NeMo input guardrails integration
│   └── nemo_output_rails.py       # Local citation, coverage, structure, and telemetry checks
├── tools/
│   ├── document_store.py          # Chroma search, BM25, and neighbor lookup
│   └── document_ingestion.py      # Document indexing and vector-store updates
├── document_processing/
│   └── document_loader.py         # PDF, DOCX, TXT, JSON, and Markdown extraction
├── embeddings/
│   └── local_model.py             # Local MiniLM embedding generation
├── memory/
│   ├── conversation_memory.py     # Local conversation history persistence
│   └── state_store.py             # SQLite summary cache and run state
├── documents/                     # User-provided source documents
└── data/
    ├── chroma/                    # Persistent ChromaDB collection
    ├── conversations.json         # Saved local conversations
    └── retrieval_trace.jsonl      # Per-run graph and retrieval telemetry
```

## How to use each component

### Use the application normally

From the project directory, start the Streamlit UI with the Python environment
where the dependencies are installed:

```powershell
python -m streamlit run app.py
```

Upload documents in the sidebar, then ask a question. The UI calls
`agents.retrieval_agent.run_rag()`, which preserves the stable public API while
delegating to `agents.graph.run_multi_agent()`.

### Use the stable Python API

```python
from agents.retrieval_agent import run_rag

result = run_rag("What is the refund policy?", history=[])
print(result["answer"])
print(result["citations"])
```

Use this entry point for integrations, scripts, tests, or replacing the
Streamlit frontend. Do not call individual agents when you need the complete
pipeline.

### Use individual agents or steps

```python
from agents.router import route_question
from agents.decomposer import decompose_question
from agents.verifier import verify_draft

route = route_question("Compare the two crop methods.")
parts = decompose_question("Compare the two crop methods.", route.route)
```

Use direct agent calls for unit tests, diagnostics, experiments, or custom
orchestration. Normal application code should use `run_rag()` so budgets,
tracing, routing, and compatibility behavior remain centralized.

### Use retrieval without answer generation

```python
from agents.retriever_worker import retrieve

retrieval = retrieve("What is the refund window?")
for chunk in retrieval.get("chunks", []):
    print(chunk["title"], chunk["text"])
```

Use the worker for evidence inspection, retrieval debugging, or a custom writer.
It returns local passages and retrieval metadata, not a final answer.

### Use deterministic curation without an LLM

```python
from agents.curator import rerank_chunks, expand_chunks

ranked, confidence = rerank_chunks(question, chunks)
context = expand_chunks(ranked, max_chars=12000)
```

Use the curator after retrieval when you need repeatable deduplication, semantic
reranking, neighbor expansion, and context sizing. The cross-encoder reranks
candidates for each question and falls back to hybrid scoring if it cannot load.

### Use document tools for indexing and search

```powershell
.\venv\Scripts\python.exe -m tools.document_ingestion
```

Use `tools.document_ingestion` to build or refresh the local index. Use
`tools.document_store` for custom document search or neighbor inspection. These
tools operate only on uploaded local documents and do not perform web search.

---

## 📚 Add Your Documents

Place source documents in the `documents/` folder at the project root, or upload them from the app's Document Library sidebar. The system supports searchable text in PDF, JSON, DOCX, TXT, and Markdown (`.md`) files up to 200 MB each; there is no page-count limit, so documents with more than 500 pages can be indexed. JSON files should contain either a string `text` field or a `pages` array. Subfolders are searched too. Scanned/image-only PDFs without extractable text are skipped.

Documents are split into 1,000-character chunks with 150 characters of overlap (up to 100 million extracted characters per file) and indexed in batches in a persistent ChromaDB collection at `data/chroma`. The `document_processing/` package handles text extraction, cleanup, structured table/record extraction, and page-aware chunking; `tools/document_ingestion.py` coordinates embeddings and storage. Structured blocks are retained in chunk metadata with their headers, row order, and cell values, then included in grounded answer context. DOCX, CSV, JSON arrays/records, and text-based PDF tables are supported. Scanned PDF OCR additionally requires the Tesseract executable installed on the host; without it, the loader reports an explicit extraction error rather than silently treating the scan as ordinary text. The project explicitly uses Chroma's lightweight local `all-MiniLM-L6-v2` embedding model for both document chunks and query text. It is much smaller than BGE-M3 and is suitable for local development. Run `python -m tools.document_ingestion` from the project root to index or refresh the collection and print its chunk/embedding counts. Research searches also synchronize the index when files are added, removed, or changed.

Image and figure records are indexed in SQLite with their existing file path,
source document, page, caption, and metadata. The database stores references,
not image bytes, so this index does not duplicate image files or add image
payloads to the database. Existing figure metadata is backfilled when the
image-asset index is first initialized.

Hybrid document retrieval merges independently ranked Chroma vector and SQLite FTS5 candidates with weighted Reciprocal Rank Fusion (RRF), avoiding direct comparison of cosine distance with query-dependent BM25 scores. Candidate depth and channel weights are set in `config.py`; each search logs separate vector, FTS5, and combined timings with query scope and matched chunk IDs/scores. SQLite reuses one initialized connection per thread for the default state database, while temporary/custom database paths remain isolated. Ingestion reconciles both indexes by chunk ID and source metadata, removing stale FTS-only rows and rebuilding missing FTS records. Run `python -m tools.retrieval_benchmark` to compare vector, FTS5, hybrid, and reranked retrieval against the labeled questions in `data/retrieval_eval_questions.json`. The benchmark writes detailed per-query results to `data/retrieval_eval_latest.json`, including source/page evidence hits, rerank latency, an RRF lexical-weight sweep, and Chroma/FTS5 index consistency. `data/retrieval_baseline.json` preserves the pre-tuning baseline.

## Agent, step, and infrastructure responsibilities

| Component | Category | Responsibility |
|---|---|---|
| Router | Agent | Classifies the question as `factual`, `multi_part`, `comparison`, `summary`, or `numeric`. |
| Decomposer | Agent-like step | Converts compound questions into 1–4 independent retrieval questions. |
| Retriever worker | Agent | Runs the existing plan → search → grade → retry LangGraph subgraph. |
| Curator | Deterministic function | Deduplicates evidence, reranks it, expands neighbors, and applies context limits. It makes no LLM calls. |
| Writer | Generation step | Produces a grounded answer without exposing chunk/source metadata and uses `<DOCUMENT_PASSAGE>` data boundaries. |
| Verifier | Agent | Checks draft claims against cited local passages and requests revision or targeted retrieval. |
| `graph.py` | Orchestrator | Wires routes, `Send` fan-out, verification loops, budgets, and state reducers. |
| ChromaDB / MiniLM | Infrastructure | Persists the local vector index and computes local embeddings. |
| Groq | Infrastructure | Hosts the configured Orchestrator, Writer, and retrieval model calls. |
| Gemini API | Summarization | Generates full-document map/reduce summaries. |
| Cohere API | Claim verification | Verifies generated claims against retrieved evidence. |

## Retrieval and verification flow

The Executor Agent resolves follow-ups from conversation history, selects the
retrieval mode, and searches local documents using hybrid Chroma/BM25 retrieval
(plus structured and visual tools when applicable). Complex comparison and
multi-part questions first receive a single Orchestrator LLM planning call;
ordinary factual questions, follow-ups, and summary requests use deterministic
planning. The Curator deduplicates and reranks retrieved evidence, then expands
neighboring passages to form the answer context.

Every answerable executor response receives deterministic citation/provenance
checks; question simplicity alone does not bypass verification. Exact
citation-backed extracts can pass without a Cohere request. Generated claims,
comparisons, and claims requiring semantic judgment are verified together in a
single Cohere request against retrieved evidence. At most one additional
targeted batch can use other already-retrieved evidence when fewer than three
claims are supported. Responses require valid claim IDs, verdicts, evidence IDs,
and exact quotes. Unsupported claims are removed or the system abstains. If the
verifier is unavailable, a generated draft is shown only when all its citations
map to retrieved passages, and it is prominently marked unverified; missing or
invalid citations still fail closed. Direct replies and clarification questions
bypass the executor verifier.
Low-scoring retrievals abstain before answer generation.

`MIN_EVIDENCE_SCORE` currently starts at `0.30`; it is a provisional floor for
the normalized reranker score, not a corpus-calibrated guarantee.

Every question receives one `run_id`. Trace records in
`data/retrieval_trace.jsonl` include the run ID, agent, sub-question where
applicable, and current LLM-call count.

Document embeddings are computed locally. Summary text is sent to Gemini;
generated answers and retrieved evidence may be sent to Cohere for semantic
verification; other model tasks use Groq. Do not add confidential
documents unless sharing their relevant text with those APIs is acceptable.

## Configuration

The main settings are in `config.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `DEFAULT_MODEL` | `qwen/qwen3.8-27b` | Writer and retrieval-worker model. |
| `SUMMARY_MODEL` | `gemini-3.5-flash-lite` | Gemini model for full-document map/reduce summaries. |
| `SUMMARY_BATCH_SIZE` | `200` | Maximum source chunks per summary map request; oversized requests are split automatically. |
| `SUMMARY_MODEL_INPUT_TOKEN_LIMIT` | `1048576` | Per-request Gemini input context limit checked before generation. |
| `GEMINI_TPM_LIMIT` | `180000` | Shared summary input-token pacing budget per minute. |
| `GEMINI_TPM_HARD_LIMIT` | `250000` | Maximum isolated summary request budget. |
| `GEMINI_TPM_WINDOW_SECONDS` | `60` | Rolling window used for Gemini summary token budgeting. |
| `GEMINI_MAX_RETRIES` | `3` | Retries for transient Gemini summary 429/503 responses. |
| `VERIFIER_MODEL` | `command-a-03-2025` | Cohere model for batched claim verification. |
| `ROUTER_MODEL` | `qwen/qwen3.8-27b` | Route classification and query planning. |
| `DECOMPOSER_MODEL` | `qwen/qwen3.8-27b` | Legacy compound-question splitting. |
| `MAX_SUB_QUESTIONS` | `4` | Maximum fan-out questions. |
| `MAX_TOTAL_LLM_CALLS` | `12` | Hard per-question budget for Groq-routed calls. |
| `MAX_PARALLEL_WORKERS` | `3` | Groq concurrency semaphore size. |
| `MAX_RETRIEVAL_ATTEMPTS` | `2` | Planner/search/grade attempts per worker. |
| `RETRIEVAL_RESULTS_PER_QUERY` | `20` | Hybrid candidates passed to answer curation. |
| `MIN_EVIDENCE_SCORE` | `0.30` | Provisional minimum normalized reranker score; not a corpus-calibrated guarantee. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Semantic cross-encoder used to rank candidate passages. |
| `ENABLE_CROSS_ENCODER` | `False` | Enables optional semantic reranking; the deterministic hybrid reranker is used by default. Set to `True` only when PyTorch and sentence-transformers work in the active environment. |
| `USE_MULTI_AGENT_RAG` | `True` | Switches between the new graph and legacy pipeline. |

## Known operational characteristics

- Multi-part questions make more model calls than factual questions because they
  use routing, decomposition, one or more retrieval workers, writing, and
  verification.
- The hard `MAX_TOTAL_LLM_CALLS` budget prevents unbounded feedback loops. Retries
  and fallback-model attempts count toward that Groq call budget. Gemini summary
  calls share a persistent token budget in the SQLite state store.
- Groq rate-limit exposure is higher for compound questions. Calls are protected
  by a shared semaphore and bounded retry/backoff logic. Explicit daily token
  quota errors skip same-model retries, try each configured fallback once, and
  surface the provider's reset guidance when no model remains available.
- The cross-encoder is disabled by default because it adds model-load latency,
  may download its model, and depends on a compatible PyTorch installation. Set
  `ENABLE_CROSS_ENCODER = True` in `config.py` only after confirming that
  `sentence-transformers` loads successfully. Otherwise, Nexus uses the
  deterministic 65/35 hybrid reranker.
- Embeddings remain local; summary text is sent to Gemini for summarization,
  answers and document evidence may be sent to Cohere for semantic
  verification, and relevant passage text may be sent to Groq for planning and writing.
  Gemini TPM accounting is shared by processes that use
  the same SQLite state database; callers on other machines or external apps
  using the same API project are not coordinated by this local database.
- Retrieval and graph traces are append-only JSON Lines in
  `data/retrieval_trace.jsonl`; they may grow over time and can be archived
  between development runs.

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Ensure Python 3.10+ is installed, then create and activate a project virtual
environment in PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Environment Configuration

Create or update your `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
# Optional; default is command-a-03-2025
VERIFIER_MODEL=command-a-03-2025
```

---

## 🖥️ Launching the Streamlit Frontend

To launch the web interface:

```bash
python -m streamlit run app.py
```

If the virtual environment is not activated, run with its Python executable:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will open in your browser at:
`http://localhost:8501`

---

## ✨ Streamlit Frontend Features

- **Executive Dashboard**: Status cards, local documents explored, and claim verification results.
- **Interactive research flow**: The Streamlit chat uses the same graph entry point
  as the CLI-compatible API and preserves conversation history.
- **Research Workspace Tabs**:
  - 📄 **Research Report**: Rendered markdown report with one-click `.md` download and citations to local document filenames.
  - 🛡️ **Fact-Check Audit**: Claim-by-claim verification breakdown with status filters (`Supported`, `Inconclusive / Refuted`), explanations, and suggested corrections.
  - 📚 **Evidence & Sources**: Explorer showing local documents, source labels, and extracted quotations.
  - 🔍 **Agent Execution Trace**: Document queries, execution time telemetry, and pipeline diagnostics.
- **Long-Term Memory Management**:
  - Switch between or create new Session IDs.
  - View conversation history for ongoing research threads.
  - One-click memory reset for the active session.
- **Configuration**:
  - Configure the Groq model in `config.py`.
  - Provide the `GROQ_API_KEY` environment variable.

---

## 💻 CLI Mode

You can still run the command-line interface anytime using:

```bash
python main.py
```
