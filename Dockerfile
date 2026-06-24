FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

COPY src/requirements.txt .

# CPU-only torch first, so we don't drag in multi-GB CUDA wheels on a GPU-less
# eval host (torch is needed only for the reranker). If the eval host has a GPU,
# swap this for the default torch wheel — the reranker auto-detects cuda.
RUN pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the reranker weights into the image. Embeddings now go through the OpenAI
# API, so the only local model is the reranker. Without baking, the first /recall
# would download ~2GB from HuggingFace at runtime (and fail if the host is offline).
# Loads on CPU during build — also fails the build early if the model id is wrong.
RUN python -c "from sentence_transformers import CrossEncoder; \
CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu', max_length=1024)"

COPY src/ .

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
