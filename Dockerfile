# ===== Stage 1: Builder =====
FROM nvidia/cuda:13.0.2-base-ubuntu24.04 AS builder

SHELL ["/bin/sh", "-exc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9.21

# Install uv and build tools in one layer
COPY --link --from=ghcr.io/astral-sh/uv:0.7.14 /uv /usr/local/bin/uv

RUN apt-get update --quiet && \
    apt-get upgrade -y && \
    apt-get install --quiet --no-install-recommends -y \
        build-essential git ca-certificates libgl1 libglib2.0-0 libusb-1.0-0-dev \
        libsndfile1 ffmpeg && \
    git config --global http.version HTTP/1.1 && \
    uv python install $PYTHON_VERSION && \
    rm -rf /var/lib/apt/lists/* /root/.cache/uv/* /root/.cache/pip/*

ENV UV_PYTHON="python$PYTHON_VERSION" \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_LINK_MODE=copy \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    PATH="/app/bin:$PATH"

WORKDIR /project

# Copy minimal files for dependency build
COPY pyproject.toml uv.lock /project

# Build Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen && \
    rm -rf /root/.cache/uv/*

# Copy and build fairseq_lib
COPY ./fairseq_lib /project/fairseq_lib
RUN uv pip install --no-cache-dir -e ./fairseq_lib -vvv && \
    rm -rf /root/.cache/pip/* && \
    # Strip compiled binaries to reduce size
    find /app -type f -name "*.so" -exec strip --strip-unneeded {} \; || true

# ===== Stage 2: Runtime =====
FROM nvidia/cuda:13.0.2-base-ubuntu24.04

SHELL ["/bin/sh", "-exc"]

ARG PYTHON_VERSION=3.9.21

# Install uv runtime and Python only (no build tools)
COPY --link --from=ghcr.io/astral-sh/uv:0.7.14 /uv /usr/local/bin/uv
RUN apt-get update --quiet && \
    apt-get install --quiet --no-install-recommends -y \
        git ca-certificates \
        libgl1 libglib2.0-0 libusb-1.0-0-dev libsndfile1 ffmpeg && \
    uv python install $PYTHON_VERSION && \
    rm -rf /var/lib/apt/lists/* /root/.cache/uv/* /root/.cache/pip/*

ENV UV_PYTHON="python$PYTHON_VERSION" \
    PATH="/app/bin:$PATH" \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH="/project/fairseq_lib:/project:${PYTHONPATH}"

WORKDIR /project

# Copy Python environment + fairseq from builder
COPY --from=builder /app /app
COPY --from=builder /project/fairseq_lib /project/fairseq_lib

# Copy only runtime project files
COPY ./.project-root /project/.project-root
COPY ./src /project/src
COPY ./scripts /project/scripts
COPY ./configs /project/configs

# Keep container alive
CMD ["sleep", "infinity"]
