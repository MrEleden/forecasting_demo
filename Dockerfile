# Multi-stage Dockerfile for ML Forecasting Portfolio
# Optimized for production deployment

# =============================================================================
# Stage 1: Base image with dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# Stage 2: Dependencies installation
# =============================================================================
FROM base as builder

# Copy requirements
COPY requirements.txt requirements-ml.txt requirements-models.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-ml.txt && \
    pip install -r requirements-models.txt

# =============================================================================
# Stage 3: Application
# =============================================================================
FROM base as application

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY projects/ ./projects/
COPY pyproject.toml ./

# Install the package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/results

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "ml_portfolio.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 4: Development image (with dev tools)
# =============================================================================
FROM application as development

# Copy dev requirements
COPY requirements-dev.txt ./

# Install dev dependencies
RUN pip install -r requirements-dev.txt

# Copy tests and config
COPY tests/ ./tests/
COPY .github/ ./.github/
COPY docs/ ./docs/

# Install pre-commit hooks
COPY .pre-commit-config.yaml ./
RUN pip install pre-commit && pre-commit install || true

CMD ["bash"]

# =============================================================================
# Stage 5: Training image
# =============================================================================
FROM application as training

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Create MLflow directory
RUN mkdir -p /app/mlruns

# Entry point for training
ENTRYPOINT ["python", "src/ml_portfolio/training/train.py"]

# =============================================================================
# Stage 6: Serving image (optimized for production)
# =============================================================================
FROM application as serving

# Install only serving dependencies
RUN pip install uvicorn[standard] gunicorn

# Non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Optimized Gunicorn + Uvicorn for production
CMD ["gunicorn", "ml_portfolio.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
