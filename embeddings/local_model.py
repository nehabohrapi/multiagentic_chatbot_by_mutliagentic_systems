from functools import lru_cache

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from config import DOCUMENT_EMBEDDING_BATCH_SIZE


@lru_cache(maxsize=1)
def get_embedding_function() -> DefaultEmbeddingFunction:
    """Return Chroma's lightweight local all-MiniLM-L6-v2 embedder."""
    return DefaultEmbeddingFunction()


@lru_cache(maxsize=1)
def _get_onnx_embedding_model() -> ONNXMiniLM_L6_V2:
    """Reuse one ONNX session and prefer CUDA when ONNX Runtime exposes it."""
    import onnxruntime

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return ONNXMiniLM_L6_V2(
            preferred_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    return ONNXMiniLM_L6_V2()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed texts with Chroma's exact ONNX MiniLM model in larger batches."""
    if not texts:
        return []
    embeddings = _get_onnx_embedding_model()._forward(
        texts,
        batch_size=DOCUMENT_EMBEDDING_BATCH_SIZE,
    )
    return [[float(value) for value in embedding] for embedding in embeddings]


@lru_cache(maxsize=256)
def embed_query(query: str) -> tuple[float, ...]:
    """Cache recent query vectors to avoid repeated local inference."""
    return tuple(float(value) for value in get_embedding_function()([query])[0])
