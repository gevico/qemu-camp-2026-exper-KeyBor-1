# SunnyHaze LeNet Assets

This directory holds generated LeNet inference assets derived from:

```text
https://github.com/SunnyHaze/LeNet5-MNIST-Pytorch
```

The upstream repository is licensed under Apache-2.0 and provides a pretrained
`model.pt`. The local export path is:

```text
python3 tools/export_sunnyhaze_lenet.py --download
```

The generated header stores weights and bias as Q8.8 `int32_t` values:

```text
real_value ~= q_value / 256
q_value = round(real_value * 256)
```

We do not need PyTorch for export because the upstream `model.pt` is a zip
archive with raw float32 tensor storages in a stable order for this model.

`images/*.jpg` stores a small set of real MNIST test-set images for visual
inspection. `mnist_samples_q8.bin` stores the same samples in a baremetal
friendly binary format. Generate both with:

```text
python3 tools/export_mnist_samples.py --count 5 \
    --blob assets/lenet/mnist_samples_q8.bin \
    --image-dir assets/lenet/images \
    --manifest assets/lenet/mnist_samples_manifest.md
```

Pixels are normalized with the same Q8.8 scale:

```text
q_pixel = round(pixel_u8 / 255 * 256)
```

The baremetal CPU reads one sample at a time from `mnist_samples_q8.bin`,
which is linked into `.rodata` via `tests/src/mnist_samples.S`, and uploads it
into the input tensor in GPU VRAM before launching the LeNet node sequence.
The JPEG files are intentionally kept as review artifacts so the test data can
be inspected without reading C arrays.

`mnist_samples_manifest.md` records the sample order, MNIST test-set index,
JPEG path, and ground-truth label. The baremetal log prints
`lenet_mnist_test_index`, `lenet_expected`, and `lenet_pred` for each sample so
the output can be matched back to a specific JPEG.

The baremetal LeNet path prints per-sample `expected` and `pred` values plus a
`lenet_correct/lenet_total` summary. This is currently a diagnostic check: it
proves that multiple real inputs flow through the runtime, and it also exposes
remaining numerical/preprocessing gaps instead of hiding them behind one
hand-picked sample.
