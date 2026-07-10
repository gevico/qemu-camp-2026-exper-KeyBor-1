#!/usr/bin/env zsh
set -euo pipefail

script_dir=${0:A:h}
image=${GPGPU_DEV_IMAGE:-qemu-gpgpu-dev:ubuntu24.04}

docker build "$@" -t "$image" -f "$script_dir/Dockerfile" "$script_dir"

echo "Built $image"
