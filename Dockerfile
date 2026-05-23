# ---- Build stage: install deps & download NLTK data ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install Python dependencies into a clean layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so the container doesn't need internet at runtime
RUN python -c "\
import nltk; \
nltk.download('punkt_tab', download_dir='/usr/local/nltk_data'); \
nltk.download('stopwords', download_dir='/usr/local/nltk_data')"


# ---- Runtime stage: lean final image ----
FROM python:3.12-slim

LABEL maintainer="Abhinaba Sarkar <https://github.com/as-repo1>"
LABEL description="SMS Spam Detector — Streamlit + MultinomialNB"

# Don't buffer Python stdout/stderr (better logging in Docker)
ENV PYTHONUNBUFFERED=1
# Tell NLTK where we put the data
ENV NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

# Copy installed packages and NLTK data from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/nltk_data /usr/local/nltk_data

# Copy application code
COPY app.py .
COPY models/ models/

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

# Streamlit config: disable telemetry, bind to all interfaces
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
