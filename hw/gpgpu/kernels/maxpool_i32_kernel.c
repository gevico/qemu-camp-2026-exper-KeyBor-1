#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUMaxPool2DArgs *args)
{
    uint32_t ow = gpgpu_thread_id_x();
    uint32_t n = gpgpu_block_id_x();
    uint32_t c = gpgpu_block_id_y();
    uint32_t oh = gpgpu_block_id_z();
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;
    uint32_t input_base = n * args->input.stride_n +
                          c * args->input.stride_c +
                          oh * args->stride_h * args->input.stride_h +
                          ow * args->stride_w;
    uint32_t output_base = n * args->output.stride_n +
                           c * args->output.stride_c +
                           oh * args->output.stride_h +
                           ow;
    int32_t max_value = input[input_base];

    for (uint32_t kh = 0; kh < args->kernel_h; ++kh) {
        uint32_t row_base = input_base + kh * args->input.stride_h;

        for (uint32_t kw = 0; kw < args->kernel_w; ++kw) {
            int32_t value = input[row_base + kw];

            if (value > max_value) {
                max_value = value;
            }
        }
    }

    output[output_base] = max_value;

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
