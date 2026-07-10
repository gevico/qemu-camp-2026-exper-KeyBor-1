FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH=/root/.cargo/bin:${PATH}

RUN sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ccache \
    clang \
    curl \
    file \
    gdb-multiarch \
    git \
    libclang-dev \
    make \
    meson \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    xz-utils \
    build-essential \
    binutils-riscv64-unknown-elf \
    gcc-riscv64-unknown-elf \
    libaio-dev \
    libattr1-dev \
    libbpf-dev \
    libbz2-dev \
    libcap-ng-dev \
    libfdt-dev \
    libglib2.0-dev \
    libgnutls28-dev \
    libpixman-1-dev \
    libsasl2-dev \
    libseccomp-dev \
    libslirp-dev \
    libssh-dev \
    liburing-dev \
    libzstd-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN riscv64-unknown-elf-gcc --version

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal \
    && cargo install bindgen-cli \
    && rustc --version \
    && cargo --version \
    && bindgen --version

ENV CCACHE_DIR=/work/.ccache
WORKDIR /work
