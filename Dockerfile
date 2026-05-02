# ─────────────────────────────────────────────────────────────────────────────
# Aegis AI Backend — CPU image (works on any machine, no GPU required)
# For GPU (Tesla P100) use: Dockerfile.gpu
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Build-time commit label (injected by deploy.sh via --build-arg)
ARG COMMIT_ID=unknown
LABEL org.opencontainers.image.revision=${COMMIT_ID}
LABEL org.opencontainers.image.description="Aegis AI Backend (CPU)"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    postgresql-client \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Stamp the commit ID into the running container so /health can expose it
ENV AEGIS_COMMIT_ID=${COMMIT_ID}

RUN mkdir -p uploads evidence models processed_videos

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-proxy-headers"]
