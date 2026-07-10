#!/usr/bin/env zsh
set -euo pipefail

image=${GPGPU_DEV_IMAGE:-qemu-gpgpu-dev:ubuntu24.04}
expected_endpoint_prefix=${GPGPU_DOCKER_ENDPOINT_PREFIX:-/Volumes/1T/DevEnv/colima}

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found in PATH" >&2
  exit 1
fi

context=$(docker context show)
endpoint=$(docker context inspect "$context" --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)

if [[ "$endpoint" != unix://$expected_endpoint_prefix/* ]]; then
  cat >&2 <<EOF
Docker context does not look like the external work environment.

Current context:  $context
Current endpoint: $endpoint
Expected prefix:  unix://$expected_endpoint_prefix/

Run cwork first, or override GPGPU_DOCKER_ENDPOINT_PREFIX if this is intentional.
EOF
  exit 1
fi

repo_root=$(git -C "${0:A:h}" rev-parse --show-toplevel)

docker image inspect "$image" >/dev/null 2>&1 || {
  echo "Image $image not found. Building it first..."
  "${0:A:h}/build-image.zsh"
}

docker run --rm -it \
  -v "$repo_root:/work" \
  -w /work \
  -e CCACHE_DIR=/work/.ccache \
  -e CARGO_HOME=/root/.cargo \
  -e RUSTUP_HOME=/root/.rustup \
  --name qemu-gpgpu-dev-shell \
  "$image" \
  bash -lc 'mkdir -p "$CCACHE_DIR"; git config --global --add safe.directory /work 2>/dev/null || true; exec bash'
