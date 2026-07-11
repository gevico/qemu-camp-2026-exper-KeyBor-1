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

`mnist_test0_q8.h` stores MNIST test-set image 0 and its label. Pixels are
normalized with the same Q8.8 scale:

```text
q_pixel = round(pixel_u8 / 255 * 256)
```
