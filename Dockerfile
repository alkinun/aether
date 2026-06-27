# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

ARG RUST_VERSION=1.89.0
ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTHON_TRAINING_DEPS="datasets huggingface-hub tqdm transformers"
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH \
    LIBTORCH_USE_PYTORCH=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        clang \
        cmake \
        git \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir ${PYTHON_TRAINING_DEPS}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    export LD_LIBRARY_PATH="$(python3 -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH}" \
    && cargo build --release -p psyche-centralized-server \
    && cp target/release/psyche-centralized-server /usr/local/bin/psyche-centralized-server

FROM python:3.12-slim-bookworm AS runtime

ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTHON_TRAINING_DEPS="datasets huggingface-hub tqdm transformers"
ENV LIBTORCH_USE_PYTORCH=1 \
    PYTHONUNBUFFERED=1 \
    RUST_LOG=info \
    TRAINING_RUN_CONFIG=config/training-run.toml \
    CONTROL_PORT=8080 \
    SERVER_PORT=39405 \
    WEB_PORT=8081

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir ${PYTHON_TRAINING_DEPS}

WORKDIR /app
COPY --from=builder /usr/local/bin/psyche-centralized-server /usr/local/bin/psyche-centralized-server
COPY config ./config
COPY scripts ./scripts

VOLUME ["/app/data", "/app/.aether-control"]

EXPOSE 39405 8080 8081

CMD ["/bin/sh", "-c", "export LD_LIBRARY_PATH=$(python3 -c \"import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\"):${LD_LIBRARY_PATH}; exec python3 scripts/training-control-dashboard.py"]
