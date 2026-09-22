ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm

FROM ${UV_IMAGE} AS uv_source

FROM ${CUDA_IMAGE}

ARG NODE_VERSION=v22.23.2
ARG NODE_ARCH=linux-x64
ARG CODEX_NPM_PACKAGE=@openai/codex@0.155.1
ARG APP_UID=1001
ARG APP_GID=1001
ARG APP_USER=runner
ARG RUST_TOOLCHAIN=stable
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_TOOL_HOME=/opt/cargo-tools
ENV CARGO_HOME=/opt/vibesim-cargo-cache
ENV DG_USE_LOCAL_VERSION=0
ENV VIBESIM_BAKED_PROJECT=/opt/vibesim-prewarm
ENV VIBESIM_BAKED_TARGET=/opt/vibesim-cache/target
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV UV_PROJECT_ENVIRONMENT=/opt/vibesim-venv
ENV UV_CACHE_DIR=/opt/vibesim-uv-cache
ENV ANALYZER_MCP_VENV=/opt/vibesim-analyzer-mcp-venv
ENV PATH=/opt/cargo-tools/bin:/opt/node/bin:${PATH}

RUN apt-get update -qq \
  && apt-get install -y -qq --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    mold \
    ninja-build \
    pkg-config \
    protobuf-compiler \
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
  && CARGO_HOME="${CARGO_TOOL_HOME}" sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
  && rm -f /tmp/rustup-init.sh \
  && CARGO_HOME="${CARGO_TOOL_HOME}" "${CARGO_TOOL_HOME}/bin/cargo" install just --locked \
  && ln -sf "${CARGO_TOOL_HOME}/bin/cargo" /usr/local/bin/cargo \
  && ln -sf "${CARGO_TOOL_HOME}/bin/just" /usr/local/bin/just \
  && ln -sf "${CARGO_TOOL_HOME}/bin/rustc" /usr/local/bin/rustc \
  && chmod -R a+rX /opt/rustup "${CARGO_TOOL_HOME}" \
  && command -v cargo \
  && command -v just \
  && command -v rustc \
  && cargo --version \
  && just --version \
  && rustc --version

RUN npm install -g --engine-strict --include=optional "${CODEX_NPM_PACKAGE}" \
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
    "${CARGO_HOME}" \
    "/opt/vibesim-cache" \
    "${VIBESIM_BAKED_PROJECT}" \
    "${ANALYZER_MCP_VENV}" \
    "/workspace" \
    "${UV_PROJECT_ENVIRONMENT}" \
    "${UV_CACHE_DIR}" \
  && chown -R "${APP_UID}:${APP_GID}" \
    "/home/${APP_USER}" \
    "${CARGO_HOME}" \
    "/opt/vibesim-cache" \
    "${VIBESIM_BAKED_PROJECT}" \
    "${ANALYZER_MCP_VENV}" \
    "/workspace" \
    "${UV_PROJECT_ENVIRONMENT}" \
    "${UV_CACHE_DIR}"

COPY --chown=${APP_UID}:${APP_GID} \
  vibesim/pyproject.toml \
  vibesim/uv.lock \
  vibesim/justfile \
  /opt/vibesim-prewarm/

ENV HOME=/home/${APP_USER}

USER ${APP_UID}:${APP_GID}

RUN cd "${VIBESIM_BAKED_PROJECT}" \
  && DG_USE_LOCAL_VERSION=0 just sync \
  && uv run python -c "import torch, triton, deep_gemm; print('prewarmed', torch.__version__)"

# The uv lock is a correctness boundary for the shared Python environment. It is
# deliberately applied after the expensive OS/toolchain layers.
ARG VIBESIM_LOCK_SHA=unknown
ENV VIBESIM_BAKED_LOCK_SHA=${VIBESIM_LOCK_SHA}

RUN uv venv "${ANALYZER_MCP_VENV}" \
  && uv pip install --python "${ANALYZER_MCP_VENV}/bin/python" "mcp==1.28.1"

# Keep the multi-GB Python/CUDA dependency layer stable when only VibeSim source
# changes. Build at /workspace so Cargo dep-info matches the runtime mount path.
ENV PYTHON=/opt/vibesim-venv/bin/python3
ENV PYO3_PYTHON=/opt/vibesim-venv/bin/python3

COPY --chown=${APP_UID}:${APP_GID} vibesim/ /workspace/

RUN cd /workspace \
  && git init -q \
  && git config user.name "VibeSim runner image" \
  && git config user.email "vibesim-runner-image@example.invalid" \
  && git add -A \
  && git commit -qm "VibeSim runner target seed" \
  && uv run python -c 'from launcher.exec import cargo_build; raise SystemExit(0 if cargo_build("release", build_analyzer=True) else 1)' \
  && release_dir=/workspace/target/release \
  && rm -f \
    "$release_dir/analyze" \
    "$release_dir/analyze.d" \
    "$release_dir/simulator" \
    "$release_dir/simulator.d" \
    "$release_dir/libsimulator.d" \
    "$release_dir/libsimulator.rlib" \
  && find "$release_dir/deps" -maxdepth 1 -type f \
    \( \
      -name 'analyze-*' -o \
      -name 'simulator-*' -o \
      -name 'libsimulator-*' -o \
      -name '*timing_kernel_derive*' -o \
      -name '*schema_derive*' \
    \) -delete \
  && find "$release_dir/.fingerprint" -mindepth 1 -maxdepth 1 -type d \
    \( \
      -name 'analyzer-*' -o \
      -name 'simulator-*' -o \
      -name 'timing-kernel-derive-*' -o \
      -name 'schema-derive-*' \
    \) -exec rm -rf -- {} + \
  && find "$release_dir/build" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'analyzer-*' -o -name 'simulator-*' \) \
    -exec rm -rf -- {} + \
  && mv /workspace/target "${VIBESIM_BAKED_TARGET}"

WORKDIR /workspace

# Keep the Claude CLI layer independent of the CUDA/Python/Cargo caches.
USER root
ARG CLAUDE_NPM_PACKAGE=@anthropic-ai/claude-code@2.1.278
RUN npm install -g --engine-strict --include=optional "${CLAUDE_NPM_PACKAGE}" \
  && command -v claude \
  && claude --version
USER ${APP_UID}:${APP_GID}

# Metadata labels last so bumping the runner/codex version does not invalidate
# the expensive apt/node/rust/prewarm layers above.
ARG RUNNER_VERSION=prebuilt-agent-runner-v12
LABEL org.vibesim.agent.runner.version="${RUNNER_VERSION}"
LABEL org.vibesim.agent.main-lock-sha="${VIBESIM_LOCK_SHA}"
