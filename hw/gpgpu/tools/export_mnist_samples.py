#!/usr/bin/env python3
"""Export MNIST test images as JPEGs plus a baremetal Q8.8 binary blob."""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import subprocess
import tempfile
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


def write_jpeg(path: Path, pixels: bytes, rows: int, cols: int, scale: int) -> None:
    cjpeg = shutil.which("cjpeg")
    if cjpeg is None:
        raise RuntimeError("cjpeg is required to export JPEG review images")

    path.parent.mkdir(parents=True, exist_ok=True)
    if scale > 1:
        data = bytearray()
        for r in range(rows):
            src_row = pixels[r * cols : (r + 1) * cols]
            scaled_row = bytearray()
            for value in src_row:
                scaled_row.extend([value] * scale)
            for _ in range(scale):
                data.extend(scaled_row)
        out_rows = rows * scale
        out_cols = cols * scale
        payload = bytes(data)
    else:
        out_rows = rows
        out_cols = cols
        payload = pixels

    with tempfile.NamedTemporaryFile(suffix=".pgm") as pgm:
        pgm.write(f"P5\n{out_cols} {out_rows}\n255\n".encode("ascii"))
        pgm.write(payload)
        pgm.flush()
        with path.open("wb") as jpg:
            subprocess.run(
                [cjpeg, "-grayscale", "-quality", "100", pgm.name],
                check=True,
                stdout=jpg,
            )


def write_q8_blob(path: Path, images: list[bytes], labels: list[int],
                  rows: int, cols: int, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<III", count, rows, cols))
        for i in range(count):
            out.write(struct.pack("<I", labels[i]))
        for i in range(count):
            for pixel in images[i]:
                out.write(struct.pack("<i", q8_pixel(pixel)))


def write_manifest(path: Path, labels: list[int], count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MNIST LeNet Sample Manifest",
        "",
        "| sample | MNIST test index | JPEG | ground truth |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(count):
        jpg = f"images/mnist_test_{i}_label_{labels[i]}.jpg"
        lines.append(f"| {i} | {i} | `{jpg}` | {labels[i]} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=5, help="number of test images to export"
    )
    parser.add_argument(
        "--blob",
        default="assets/lenet/mnist_samples_q8.bin",
        help="generated baremetal Q8.8 binary blob path",
    )
    parser.add_argument(
        "--image-dir",
        default="assets/lenet/images",
        help="directory for generated JPEG review images",
    )
    parser.add_argument(
        "--manifest",
        default="assets/lenet/mnist_samples_manifest.md",
        help="generated sample manifest path",
    )
    parser.add_argument(
        "--jpeg-scale",
        type=int,
        default=10,
        help="nearest-neighbor scale factor for JPEG review images",
    )
    args = parser.parse_args()

    rows, cols, images = parse_images(fetch(IMAGES_URL))
    labels = parse_labels(fetch(LABELS_URL))
    count = min(args.count, len(images), len(labels))

    image_dir = Path(args.image_dir)
    for i in range(count):
        write_jpeg(
            image_dir / f"mnist_test_{i}_label_{labels[i]}.jpg",
            images[i],
            rows,
            cols,
            args.jpeg_scale,
        )
    write_q8_blob(Path(args.blob), images, labels, rows, cols, count)
    write_manifest(Path(args.manifest), labels, count)


if __name__ == "__main__":
    main()
