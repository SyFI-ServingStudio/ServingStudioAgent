ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm

FROM ${UV_IMAGE} AS uv_source

FROM ${CUDA_IMAGE}

ARG NODE_VERSION=v20.18.1
ARG NODE_ARCH=linux-x64
ARG CODEX_NPM_PACKAGE=@openai/codex@0.125.0
ARG APP_UID=1001
ARG APP_GID=1001
ARG APP_USER=kanzhu
ARG RUST_TOOLCHAIN=stable
ARG RUNNER_VERSION=prebuilt-codex-runner-v5
ARG MLSIM_LOCK_SHA=unknown
ARG DEBIAN_FRONTEND=noninteractive

LABEL org.mlsim.ui.codex-runner.version="${RUNNER_VERSION}"
LABEL org.mlsim.ui.main-lock-sha="${MLSIM_LOCK_SHA}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV DG_USE_LOCAL_VERSION=0
ENV MLSIM_BAKED_LOCK_SHA=${MLSIM_LOCK_SHA}
ENV MLSIM_BAKED_PROJECT=/opt/mlsim-prewarm
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV UV_PROJECT_ENVIRONMENT=/opt/mlsim-venv
ENV UV_CACHE_DIR=/opt/mlsim-uv-cache
ENV PATH=/opt/cargo/bin:/opt/node/bin:${PATH}

RUN apt-get update -qq \
  && apt-get install -y -qq --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    python-is-python3 \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    xz-utils \
  && rm -rf /var/lib/apt/lists/*

COPY --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

RUN cd /tmp \
  && curl -fsSL --retry 5 --retry-delay 5 \
    -o node.tgz \
    "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-${NODE_ARCH}.tar.gz" \
  && mkdir -p /opt/node \
  && tar -xzf node.tgz -C /opt/node --strip-components=1 \
  && ln -sf /opt/node/bin/node /usr/local/bin/node \
  && ln -sf /opt/node/bin/npm /usr/local/bin/npm \
  && ln -sf /opt/node/bin/npx /usr/local/bin/npx \
  && rm -f node.tgz

RUN curl -fsSL --retry 5 --retry-delay 5 https://sh.rustup.rs -o /tmp/rustup-init.sh \
  && sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
  && rm -f /tmp/rustup-init.sh \
  && cargo install just --locked \
  && ln -sf /opt/cargo/bin/cargo /usr/local/bin/cargo \
  && ln -sf /opt/cargo/bin/just /usr/local/bin/just \
  && ln -sf /opt/cargo/bin/rustc /usr/local/bin/rustc \
  && chmod -R a+rX /opt/rustup /opt/cargo \
  && command -v cargo \
  && command -v just \
  && command -v rustc \
  && cargo --version \
  && just --version \
  && rustc --version

RUN npm install -g --include=optional "${CODEX_NPM_PACKAGE}" \
  && codex_bin="$(npm prefix -g)/bin/codex" \
  && test -x "$codex_bin" \
  && ln -sf "$codex_bin" /usr/local/bin/codex \
  && command -v bash \
  && command -v cargo \
  && command -v git \
  && command -v just \
  && command -v node \
  && command -v npm \
  && command -v nvcc \
  && command -v python \
  && command -v python3 \
  && command -v rustc \
  && command -v uv \
  && command -v codex \
  && nvcc --version \
  && python --version \
  && uv --version

RUN if ! getent group "${APP_GID}" >/dev/null 2>&1; then \
      groupadd -g "${APP_GID}" "${APP_USER}"; \
    fi \
  && if ! getent passwd "${APP_UID}" >/dev/null 2>&1; then \
      useradd -m -u "${APP_UID}" -g "${APP_GID}" -s /bin/bash -d "/home/${APP_USER}" "${APP_USER}"; \
    fi \
  && mkdir -p \
    "/home/${APP_USER}/.cache" \
    "/home/${APP_USER}/.local" \
    "/home/${APP_USER}/.npm" \
    "/workspace" \
    "${MLSIM_BAKED_PROJECT}" \
    "${UV_PROJECT_ENVIRONMENT}" \
    "${UV_CACHE_DIR}" \
  && chown -R "${APP_UID}:${APP_GID}" \
    "/home/${APP_USER}" \
    "/workspace" \
    "${MLSIM_BAKED_PROJECT}" \
    "${UV_PROJECT_ENVIRONMENT}" \
    "${UV_CACHE_DIR}"

COPY --chown=${APP_UID}:${APP_GID} mlsim/ /opt/mlsim-prewarm/

ENV HOME=/home/${APP_USER}

USER ${APP_UID}:${APP_GID}

RUN cd "${MLSIM_BAKED_PROJECT}" \
  && DG_USE_LOCAL_VERSION=0 just sync \
  && uv run python -c "import torch, triton, deep_gemm; print('prewarmed', torch.__version__)"

WORKDIR /workspace
