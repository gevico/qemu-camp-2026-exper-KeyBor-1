#!/usr/bin/env python3
"""
Export SunnyHaze LeNet-5 PyTorch weights to a freestanding Q8.8 C header.

The source model is a small PyTorch zip archive. We avoid importing torch by
reading the raw float32 storages directly from archive/data/{0..9}; their order
and shapes are fixed by the repository's LeNet_5 definition.
"""

from __future__ import annotations

import argparse
import math
import struct
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "SunnyHaze/LeNet5-MNIST-Pytorch/main/model.pt"
)
DEFAULT_OUTPUT = "assets/lenet/sunnyhaze_lenet_q8_weights.h"
Q8_SCALE = 256

TENSORS = [
    ("conv1_weight", "archive/data/0", (6, 1, 5, 5)),
    ("conv1_bias", "archive/data/1", (6,)),
    ("conv2_weight", "archive/data/2", (16, 6, 5, 5)),
    ("conv2_bias", "archive/data/3", (16,)),
    ("fc1_weight", "archive/data/4", (120, 400)),
    ("fc1_bias", "archive/data/5", (120,)),
    ("fc2_weight", "archive/data/6", (84, 120)),
    ("fc2_bias", "archive/data/7", (84,)),
    ("fc3_weight", "archive/data/8", (10, 84)),
    ("fc3_bias", "archive/data/9", (10,)),
]


def numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= dim
    return value


def quantize_q8(value: float) -> int:
    scaled = value * Q8_SCALE
    if scaled >= 0:
        return int(math.floor(scaled + 0.5))
    return int(math.ceil(scaled - 0.5))


def read_f32_storage(zf: zipfile.ZipFile, member: str, shape: tuple[int, ...]) -> list[float]:
    data = zf.read(member)
    expected = numel(shape) * 4
    if len(data) != expected:
        raise ValueError(f"{member}: expected {expected} bytes, got {len(data)}")
    return list(struct.unpack("<" + "f" * numel(shape), data))


def c_array(name: str, values: list[int]) -> str:
    lines = [f"static const int32_t {name}[] = {{"]
    for i in range(0, len(values), 8):
        chunk = ", ".join(str(v) for v in values[i : i + 8])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)


def shape_macro(name: str, shape: tuple[int, ...]) -> str:
    suffixes = ["N", "C", "H", "W"] if len(shape) == 4 else ["N", "C"]
    if len(shape) == 1:
        suffixes = ["LEN"]
    return "\n".join(
        f"#define SUNNYHAZE_LENET_{name.upper()}_{suffix} {dim}"
        for suffix, dim in zip(suffixes, shape)
    )


def download_model(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())


def export_header(model: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_arrays: list[str] = []
    rendered_shapes: list[str] = []

    with zipfile.ZipFile(model) as zf:
        for name, member, shape in TENSORS:
            values = read_f32_storage(zf, member, shape)
            qvalues = [quantize_q8(v) for v in values]
            rendered_shapes.append(shape_macro(name, shape))
            rendered_arrays.append(c_array(f"sunnyhaze_lenet_{name}_q8", qvalues))

    body = "\n\n".join(rendered_shapes + rendered_arrays)
    output.write_text(
        f"""\
/*
 * Generated from SunnyHaze/LeNet5-MNIST-Pytorch model.pt.
 * Source: {DEFAULT_MODEL_URL}
 * License: Apache-2.0, see assets/lenet/README.md.
 *
 * Quantization: Q8.8 stored in int32_t.
 * real_value ~= q_value / {Q8_SCALE}
 */

#ifndef GPGPU_SUNNYHAZE_LENET_Q8_WEIGHTS_H
#define GPGPU_SUNNYHAZE_LENET_Q8_WEIGHTS_H

#include <stdint.h>

#define SUNNYHAZE_LENET_Q8_SHIFT 8
#define SUNNYHAZE_LENET_Q8_SCALE {Q8_SCALE}

{body}

#endif /* GPGPU_SUNNYHAZE_LENET_Q8_WEIGHTS_H */
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="assets/lenet/sunnyhaze_model.pt")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    model = Path(args.model)
    output = Path(args.output)

    if args.download or not model.exists():
        download_model(args.url, model)
    export_header(model, output)


if __name__ == "__main__":
    main()
