# Multi-Architecture Cross-Platform Dockerfile
# Supports linux/amd64 (CUDA-enabled datacenter/workstations) and linux/arm64 (macOS development fallbacks)

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory inside container
WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Multi-Architecture installation logic:
# If target is amd64, we install PyTorch compiled with CUDA 12.1.
# If target is arm64 (e.g. macOS Docker Desktop), we install standard PyTorch.
ARG TARGETPLATFORM
RUN echo "Building for target platform: ${TARGETPLATFORM}" && \
    if [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu121 && \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir torch && \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the rest of the project source code
COPY . .

# Default runtime parameters (can be overridden at runtime)
ENV DEVICE=cpu
ENV MODEL_ID="MaziyarPanahi/Meta-Llama-3-8B-Instruct-AWQ"
ENV PYTHONUNBUFFERED=1

# Verification entrypoint to output system capabilities on container start
CMD ["python", "-c", "import torch; print('========================================'); print('SYSTEM DIAGNOSTICS:'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'MPS available (Apple Silicon): {torch.backends.mps.is_available()}'); print('========================================')"]
