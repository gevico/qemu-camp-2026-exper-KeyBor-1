# GPGPU Docker Dev Environment

This environment is intended for the external-disk `work` Colima profile.

The image prepares:

- Ubuntu 24.04 QEMU build dependencies for `riscv64-softmmu`.
- RISC-V bare-metal toolchain from Ubuntu packages.
- Rust toolchain plus `bindgen-cli`.

From the QEMU repository root:

```zsh
cwork
hw/gpgpu/dev/build-image.zsh
hw/gpgpu/dev/shell.zsh
```

Inside the container, `/work` is the mounted QEMU repository root.

Useful commands inside the container:

```bash
./configure --target-list=riscv64-softmmu --disable-werror
make -j"$(nproc)"
make -C hw/gpgpu/baremetal
```

`dev/shell.zsh` checks that the active Docker endpoint is under
`/Volumes/1T/DevEnv/colima` before launching the container, so it should not
accidentally use the local `mini` profile.
