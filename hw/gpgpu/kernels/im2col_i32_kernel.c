#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUIm2ColArgs *args)
{
    uint32_t kw = gpgpu_thread_id_x();
    uint32_t kh = gpgpu_thread_id_y();
    uint32_t ic = gpgpu_thread_id_z();
    uint32_t n = gpgpu_block_id_x();
    uint32_t oh = gpgpu_block_id_y();
    uint32_t ow = gpgpu_block_id_z();
    int32_t ih = (int32_t)(oh * args->stride_h + kh) -
                 (int32_t)args->pad_h;
    int32_t iw = (int32_t)(ow * args->stride_w + kw) -
                 (int32_t)args->pad_w;
    uint32_t m = n * args->out_h * args->out_w + oh * args->out_w + ow;
    uint32_t k = ic * args->kernel_h * args->kernel_w +
                 kh * args->kernel_w + kw;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;
    int32_t value = 0;

    if (ih >= 0 && iw >= 0 &&
        (uint32_t)ih < args->input.h && (uint32_t)iw < args->input.w) {
        value = input[n * args->input.stride_n +
                      ic * args->input.stride_c +
                      (uint32_t)ih * args->input.stride_h +
                      (uint32_t)iw * args->input.stride_w];
    }

    output[m * args->output.c + k] = value;

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
