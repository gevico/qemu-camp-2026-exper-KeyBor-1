#!/usr/bin/env python3
"""Export a few MNIST test images as Q8.8 C arrays."""

from __future__ import annotations

import argparse
import gzip
import struct
import urllib.request
from pathlib import Path


IMAGES_URL = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
    "t10k-images-idx3-ubyte.gz"
)
LABELS_URL = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
    "t10k-labels-idx1-ubyte.gz"
)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_images(blob: bytes) -> tuple[int, int, list[bytes]]:
    raw = gzip.decompress(blob)
    magic, count, rows, cols = struct.unpack_from(">IIII", raw, 0)
    if magic != 2051:
        raise ValueError(f"unexpected image magic {magic}")
    image_size = rows * cols
    images = [
        raw[16 + i * image_size : 16 + (i + 1) * image_size]
        for i in range(count)
    ]
    return rows, cols, images


def parse_labels(blob: bytes) -> list[int]:
    raw = gzip.decompress(blob)
    magic, count = struct.unpack_from(">II", raw, 0)
    if magic != 2049:
        raise ValueError(f"unexpected label magic {magic}")
    return list(raw[8 : 8 + count])


def q8_pixel(pixel: int) -> int:
    return int(round(pixel * 256 / 255))


def emit_array(values: list[int], indent: str = "    ") -> list[str]:
    lines = []
    for i in range(0, len(values), 14):
        chunk = values[i : i + 14]
        lines.append(indent + ", ".join(str(v) for v in chunk) + ",")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=5, help="number of test images to export"
    )
    parser.add_argument(
        "--output",
        default="assets/lenet/mnist_samples_q8.h",
        help="generated C header path",
    )
    args = parser.parse_args()

    rows, cols, images = parse_images(fetch(IMAGES_URL))
    labels = parse_labels(fetch(LABELS_URL))
    count = min(args.count, len(images), len(labels))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "/* Generated from MNIST test images, Q8.8 int32 pixels. */",
        "#ifndef GPGPU_MNIST_SAMPLES_Q8_H",
        "#define GPGPU_MNIST_SAMPLES_Q8_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define MNIST_SAMPLE_COUNT {count}",
        "#define MNIST_SAMPLE_N 1",
        "#define MNIST_SAMPLE_C 1",
        f"#define MNIST_SAMPLE_H {rows}",
        f"#define MNIST_SAMPLE_W {cols}",
        "",
        "static const uint32_t mnist_sample_labels[MNIST_SAMPLE_COUNT] = {",
        "    " + ", ".join(str(labels[i]) for i in range(count)) + ",",
        "};",
        "",
        "static const int32_t mnist_samples_q8[MNIST_SAMPLE_COUNT]"
        "[MNIST_SAMPLE_C * MNIST_SAMPLE_H * MNIST_SAMPLE_W] = {",
    ]

    for i in range(count):
        lines.append("    {")
        lines.extend(emit_array([q8_pixel(p) for p in images[i]], "        "))
        lines.append("    },")

    lines.extend(
        [
            "};",
            "",
            "#endif /* GPGPU_MNIST_SAMPLES_Q8_H */",
        ]
    )
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
