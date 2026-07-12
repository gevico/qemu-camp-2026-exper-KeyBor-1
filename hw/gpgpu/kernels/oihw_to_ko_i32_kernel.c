#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUOihwToKoArgs *args)
{
    uint32_t kw = gpgpu_thread_id_x();
    uint32_t oc = gpgpu_block_id_x();
    uint32_t ic = gpgpu_block_id_y();
    uint32_t kh = gpgpu_block_id_z();
    uint32_t k = ic * args->kernel_h * args->kernel_w +
                 kh * args->kernel_w + kw;
    uint32_t input_offset = oc * args->input.stride_n +
                            ic * args->input.stride_c +
                            kh * args->input.stride_h +
                            kw * args->input.stride_w;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;

    output[k * args->out_channels + oc] = input[input_offset];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
