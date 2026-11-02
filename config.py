import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# BASE DIRECTORIES & FILE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION_NAME = "research_documents_minilm"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DOCUMENT_CHUNK_SIZE = 1000
DOCUMENT_CHUNK_OVERLAP = 150
MAX_DOCUMENT_SEARCH_DISTANCE = 0.7
RETRIEVAL_CANDIDATE_MULTIPLIER = 8
RETRIEVAL_MIN_CANDIDATES = 40
HYBRID_RRF_K = 60
# The labeled corpus favored FTS5 ranks; retain a smaller vector contribution.
HYBRID_VECTOR_RRF_WEIGHT = 1.0
HYBRID_LEXICAL_RRF_WEIGHT = 12.0

EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ENABLE_IMAGE_EXTRACTION = True

CONVERSATIONS_DB_FILE = DATA_DIR / "conversations.sqlite3"
CONVERSATIONS_FILE = CONVERSATIONS_DB_FILE
RETRIEVAL_TRACE_FILE = DATA_DIR / "retrieval_trace.jsonl"
STATE_DB_FILE = DATA_DIR / "nexus_state.sqlite3"

# ==============================================================================
# MODEL CONFIGURATIONS & RATE LIMIT RESILIENCE
# ==============================================================================
DEFAULT_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ROUTER_MODEL = "qwen/qwen3.8-27b"
DECOMPOSER_MODEL = "qwen/qwen3.8-27b"
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "command-a-03-2025")
VERIFIER_BATCH_SIZE = int(os.getenv("VERIFIER_BATCH_SIZE", "10"))
VERIFIER_SUMMARY_BATCH_SIZE = int(os.getenv("VERIFIER_SUMMARY_BATCH_SIZE", "20"))
VERIFIER_SUMMARY_LARGE_BATCH_SIZE = int(os.getenv("VERIFIER_SUMMARY_LARGE_BATCH_SIZE", "30"))
VERIFIER_MAX_CONCURRENCY = int(os.getenv("VERIFIER_MAX_CONCURRENCY", "2"))
WRITER_MODEL = DEFAULT_MODEL
VISION_MODEL = DEFAULT_MODEL
LLM_MAX_RETRIES = 3
LLM_INITIAL_BACKOFF_SECONDS = 2.0
LLM_TEMPERATURE = 0.2

# ==============================================================================
# ORCHESTRATION HARD LIMITS (Configurable Constants)
# ==============================================================================
MAX_REVIEWER_LOOPS = 2
MAX_FACT_CHECK_LOOPS = 1
MAX_SEARCHES_PER_QUESTION = 8
MAX_PAGE_FETCHES_PER_QUESTION = 10
# Groq request budget; Gemini summaries use the separate TPM limiter above.
MAX_RETRIEVAL_ATTEMPTS = 2
RETRIEVAL_RESULTS_PER_QUERY = 20
MAX_RETRIEVAL_QUERIES = 2
SUMMARY_MAX_CONTEXT_CHARS = 24000
SUMMARY_MODEL = "gemini-3.5-flash-lite"
SUMMARY_BATCH_SIZE = 200
SUMMARY_MODEL_INPUT_TOKEN_LIMIT = 1_048_576
GEMINI_TPM_LIMIT = 180_000
GEMINI_TPM_HARD_LIMIT = 250_000
GEMINI_TPM_WINDOW_SECONDS = 60
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_SECONDS = 5
SUMMARY_MAP_MAX_OUTPUT_TOKENS = 4096
SUMMARY_FINAL_MAX_OUTPUT_TOKENS = 8192
SUMMARY_RELEVANT_CHUNKS_PER_QUERY = 12
SUMMARY_MAX_RELEVANT_CHUNKS = 48
MAX_SUB_QUESTIONS = 4
MAX_VERIFY_ROUNDS = 3
MAX_TOTAL_LLM_CALLS = 12
MAX_PARALLEL_WORKERS = 3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLE_CROSS_ENCODER = False
USE_MULTI_AGENT_RAG = True
MIN_EVIDENCE_SCORE = 0.30
PER_TOOL_TIMEOUT_SECONDS = 15
TOTAL_TIME_BUDGET_SECONDS = 90

# ==============================================================================
# SECURITY, NETWORKING & FETCH LIMITS
# ==============================================================================
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB download cap
MAX_EXTRACTED_CONTENT_LENGTH = 16000     # 16k chars text cap
MAX_DOCUMENT_BYTES = 200 * 1024 * 1024
MAX_INDEXED_DOCUMENT_LENGTH = 100 * 1024 * 1024
DOCUMENT_EMBEDDING_BATCH_SIZE = 128
CHROMA_UPSERT_BATCH_SIZE = 128
MAX_VISION_IMAGES = 3
MAX_VISION_IMAGE_BYTES = 4 * 1024 * 1024
MAX_EXTRACTED_VISUALS_PER_DOCUMENT = 30
DEFAULT_REQUEST_TIMEOUT = 12             # Seconds per HTTP request

# Blocked loopback and private IPv4 / IPv6 addresses
BLOCKED_IP_NETWORKS = [
    "127.0.0.0/8",       # IPv4 loopback
    "10.0.0.0/8",        # RFC 1918 private
    "172.16.0.0/12",     # RFC 1918 private
    "192.168.0.0/16",    # RFC 1918 private
    "169.254.0.0/16",    # IPv4 link-local
    "::1/128",           # IPv6 loopback
    "fe80::/10",         # IPv6 link-local
    "fc00::/7",          # IPv6 unique local
    "0.0.0.0/8",         # Current network
]

# Configurable list of blocked / low-quality domains (essay mills, paper writing scams)
BLOCKED_OR_LOW_QUALITY_DOMAINS = [
    "essayhub.com",
    "buy-cheap-essay-fast.com",
    "customessaymeister.com",
    "paperhelp.org",
    "grademiners.com",
    "essaypro.com",
    "edubirdie.com",
    "speedypaper.com",
    "99papers.com",
    "writepaperfor.me",
]

# ==============================================================================
# SOURCE QUALITY SCORING RUBRIC (Weights summing to 100)
# ==============================================================================
SOURCE_SCORING_WEIGHTS = {
    "relevance": 30,           # Relevance to the specific research question
    "authority": 25,           # Official (.gov, .edu, .ac.uk, primary publisher)
    "evidence_quality": 20,    # Specific, verifiable data vs vague opinions
    "recency": 15,             # Recency / fresh updates
    "corroboration": 10,       # Backed by other independent sources
}

OFFICIAL_ACADEMIC_TLDS = {".edu", ".ac.uk", ".edu.au", ".edu.in", ".ac.in", ".gov", ".gov.in", ".gov.uk"}

TOP_UNIVERSITIES = {
    "mit.edu", "ox.ac.uk", "cam.ac.uk", "harvard.edu", "stanford.edu",
    "berkeley.edu", "columbia.edu", "princeton.edu", "yale.edu", "cmu.edu",
    "imperial.ac.uk", "ucl.ac.uk", "ed.ac.uk", "utoronto.ca", "nus.edu.sg",
    "iitb.ac.in", "iitd.ac.in", "iisc.ac.in", "ugc.gov.in", "turnitin.com"
}

COMMERCIAL_ESSAY_INDICATORS = {
    "essay", "paper", "write", "buy", "service", "cheap", "fast",
    "discount", "order", "homework", "assignment-help", "plagiarism-remover"
}

POLICY_KEYWORDS = {
    "policy", "regulation", "guideline", "code", "integrity", "handbook",
    "statute", "ordinance", "procedure", "standard", "syllabus"
}
