#include "test_common.h"
#include "../../../assets/lenet/mnist_samples_q8.h"
#include "../../../assets/lenet/sunnyhaze_lenet_q8_weights.h"

int run_lenet_smoke(GPGPURuntimeDevice *dev)
{
    uint32_t im2col_kernel_addr;
    uint32_t oihw_to_ko_kernel_addr;
    uint32_t matmul_partial_kernel_addr;
    uint32_t matmul_reduce_kernel_addr;
    uint32_t layout_kernel_addr;
    uint32_t relu_kernel_addr;
    uint32_t pool_kernel_addr;
    uint32_t lenet_input_addr;
    uint32_t lenet_conv1_weight_addr;
    uint32_t lenet_conv1_bias_addr;
    uint32_t lenet_conv1_col_addr;
    uint32_t lenet_conv1_ko_addr;
    uint32_t lenet_conv1_partial_addr;
    uint32_t lenet_conv1_mo_addr;
    uint32_t lenet_conv1_nchw_addr;
    uint32_t lenet_pool1_addr;
    uint32_t lenet_conv2_weight_addr;
    uint32_t lenet_conv2_bias_addr;
    uint32_t lenet_conv2_col_addr;
    uint32_t lenet_conv2_ko_addr;
    uint32_t lenet_conv2_partial_addr;
    uint32_t lenet_conv2_mo_addr;
    uint32_t lenet_conv2_nchw_addr;
    uint32_t lenet_pool2_addr;
    uint32_t lenet_fc1_weight_ko_addr;
    uint32_t lenet_fc1_bias_addr;
    uint32_t lenet_fc1_partial_addr;
    uint32_t lenet_fc1_out_addr;
    uint32_t lenet_fc2_weight_ko_addr;
    uint32_t lenet_fc2_bias_addr;
    uint32_t lenet_fc2_partial_addr;
    uint32_t lenet_fc2_out_addr;
    uint32_t lenet_fc3_weight_ko_addr;
    uint32_t lenet_fc3_bias_addr;
    uint32_t lenet_fc3_partial_addr;
    uint32_t lenet_logits_addr;
    uint32_t lenet_args_addr[LENET_NODE_MAX];
    GPGPUNodeDesc lenet_nodes[LENET_NODE_MAX];
    uint32_t lenet_node_count = 0;
    int32_t lenet_logits[LENET_FC3_OUT];
    uint32_t lenet_pred = 0;
    uint32_t lenet_correct = 0;
    GPGPUIm2ColArgs lenet_im2col1_args;
    GPGPUOihwToKoArgs lenet_oihw1_args;
    GPGPUMatmulPartialArgs lenet_conv1_partial_args;
    GPGPUMatmulReduceArgs lenet_conv1_reduce_args;
    GPGPUMoToNchwArgs lenet_mo1_args;
    GPGPUReluArgs lenet_relu1_args;
    GPGPUMaxPool2DArgs lenet_pool1_args;
    GPGPUIm2ColArgs lenet_im2col2_args;
    GPGPUOihwToKoArgs lenet_oihw2_args;
    GPGPUMatmulPartialArgs lenet_conv2_partial_args;
    GPGPUMatmulReduceArgs lenet_conv2_reduce_args;
    GPGPUMoToNchwArgs lenet_mo2_args;
    GPGPUReluArgs lenet_relu2_args;
    GPGPUMaxPool2DArgs lenet_pool2_args;
    GPGPUMatmulPartialArgs lenet_fc1_partial_args;
    GPGPUMatmulReduceArgs lenet_fc1_reduce_args;
    GPGPUReluArgs lenet_relu3_args;
    GPGPUMatmulPartialArgs lenet_fc2_partial_args;
    GPGPUMatmulReduceArgs lenet_fc2_reduce_args;
    GPGPUReluArgs lenet_relu4_args;
    GPGPUMatmulPartialArgs lenet_fc3_partial_args;
    GPGPUMatmulReduceArgs lenet_fc3_reduce_args;
    int ret;

    ret = gpgpu_upload_kernel(dev, &im2col_kernel_addr,
                              im2col_i32_kernel_code,
                              sizeof(im2col_i32_kernel_code) /
                              sizeof(im2col_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &oihw_to_ko_kernel_addr,
                              oihw_to_ko_i32_kernel_code,
                              sizeof(oihw_to_ko_i32_kernel_code) /
                              sizeof(oihw_to_ko_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &matmul_partial_kernel_addr,
                              matmul_partial_i32_kernel_code,
                              sizeof(matmul_partial_i32_kernel_code) /
                              sizeof(matmul_partial_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &matmul_reduce_kernel_addr,
                              matmul_reduce_i32_kernel_code,
                              sizeof(matmul_reduce_i32_kernel_code) /
                              sizeof(matmul_reduce_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &layout_kernel_addr,
                              mo_to_nchw_i32_kernel_code,
                              sizeof(mo_to_nchw_i32_kernel_code) /
                              sizeof(mo_to_nchw_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &relu_kernel_addr,
                              relu_i32_kernel_code,
                              sizeof(relu_i32_kernel_code) /
                              sizeof(relu_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &pool_kernel_addr,
                              maxpool_i32_kernel_code,
                              sizeof(maxpool_i32_kernel_code) /
                              sizeof(maxpool_i32_kernel_code[0]));
    if (ret < 0) return ret;

    ret = alloc_i32_array(dev, &lenet_input_addr,
                          LENET_N * LENET_IN_C * LENET_IN_H * LENET_IN_W);
    if (ret < 0) return ret;
    ret = upload_i32_array(dev, &lenet_conv1_weight_addr,
                           sunnyhaze_lenet_conv1_weight_q8,
                           LENET_CONV1_O * LENET_IN_C *
                           LENET_CONV1_KH * LENET_CONV1_KW);
    if (ret < 0) {
        report_ret("upload(conv1_weight)", ret);
        return ret;
    }
    ret = upload_i32_array(dev, &lenet_conv1_bias_addr,
                           sunnyhaze_lenet_conv1_bias_q8, LENET_CONV1_O);
    if (ret < 0) {
        report_ret("upload(conv1_bias)", ret);
        return ret;
    }
    ret = upload_i32_array(dev, &lenet_conv2_weight_addr,
                           sunnyhaze_lenet_conv2_weight_q8,
                           LENET_CONV2_O * LENET_CONV1_O *
                           LENET_CONV2_KH * LENET_CONV2_KW);
    if (ret < 0) {
        report_ret("upload(conv2_weight)", ret);
        return ret;
    }
    ret = upload_i32_array(dev, &lenet_conv2_bias_addr,
                           sunnyhaze_lenet_conv2_bias_q8, LENET_CONV2_O);
    if (ret < 0) {
        report_ret("upload(conv2_bias)", ret);
        return ret;
    }
    ret = upload_linear_weight_ko(dev, &lenet_fc1_weight_ko_addr,
                                  sunnyhaze_lenet_fc1_weight_q8,
                                  LENET_FC1_OUT, LENET_FC1_IN);
    if (ret < 0) {
        report_ret("upload(fc1_weight_ko)", ret);
        return ret;
    }
    ret = upload_i32_array(dev, &lenet_fc1_bias_addr,
                           sunnyhaze_lenet_fc1_bias_q8, LENET_FC1_OUT);
    if (ret < 0) {
        report_ret("upload(fc1_bias)", ret);
        return ret;
    }
    ret = upload_linear_weight_ko(dev, &lenet_fc2_weight_ko_addr,
                                  sunnyhaze_lenet_fc2_weight_q8,
                                  LENET_FC2_OUT, LENET_FC1_OUT);
    if (ret < 0) {
        report_ret("upload(fc2_weight_ko)", ret);
        return ret;
    }
    ret = upload_i32_array(dev, &lenet_fc2_bias_addr,
                           sunnyhaze_lenet_fc2_bias_q8, LENET_FC2_OUT);
    if (ret < 0) {
        report_ret("upload(fc2_bias)", ret);
        return ret;
    }
    ret = upload_linear_weight_ko(dev, &lenet_fc3_weight_ko_addr,
                                  sunnyhaze_lenet_fc3_weight_q8,
                                  LENET_FC3_OUT, LENET_FC2_OUT);
    if (ret < 0) {
        report_ret("upload(fc3_weight_ko)", ret);
        return ret;
    }
    ret = upload_i32_array(dev, &lenet_fc3_bias_addr,
                           sunnyhaze_lenet_fc3_bias_q8, LENET_FC3_OUT);
    if (ret < 0) {
        report_ret("upload(fc3_bias)", ret);
        return ret;
    }

    ret = alloc_i32_array(dev, &lenet_conv1_col_addr,
                          LENET_CONV1_M * LENET_CONV1_K);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv1_ko_addr,
                          LENET_CONV1_K * LENET_CONV1_O);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv1_partial_addr,
                          LENET_CONV1_M * LENET_CONV1_O * LENET_CONV1_K);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv1_mo_addr,
                          LENET_CONV1_M * LENET_CONV1_O);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv1_nchw_addr,
                          LENET_N * LENET_CONV1_O *
                          LENET_CONV1_OUT_H * LENET_CONV1_OUT_W);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_pool1_addr,
                          LENET_N * LENET_CONV1_O *
                          LENET_POOL1_H * LENET_POOL1_W);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv2_col_addr,
                          LENET_CONV2_M * LENET_CONV2_K);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv2_ko_addr,
                          LENET_CONV2_K * LENET_CONV2_O);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv2_partial_addr,
                          LENET_CONV2_M * LENET_CONV2_O * LENET_CONV2_K);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv2_mo_addr,
                          LENET_CONV2_M * LENET_CONV2_O);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_conv2_nchw_addr,
                          LENET_N * LENET_CONV2_O *
                          LENET_CONV2_OUT_H * LENET_CONV2_OUT_W);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_pool2_addr, LENET_FC1_IN);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_fc1_partial_addr,
                          LENET_FC1_OUT * LENET_FC1_IN);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_fc1_out_addr, LENET_FC1_OUT);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_fc2_partial_addr,
                          LENET_FC2_OUT * LENET_FC1_OUT);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_fc2_out_addr, LENET_FC2_OUT);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_fc3_partial_addr,
                          LENET_FC3_OUT * LENET_FC2_OUT);
    if (ret < 0) return ret;
    ret = alloc_i32_array(dev, &lenet_logits_addr, LENET_FC3_OUT);
    if (ret < 0) return ret;

    lenet_im2col1_args = (GPGPUIm2ColArgs) {
        .input = gpgpu_tensor_make_nchw_i32(lenet_input_addr, LENET_N,
                                            LENET_IN_C, LENET_IN_H,
                                            LENET_IN_W),
        .output = gpgpu_tensor_make_mk_i32(lenet_conv1_col_addr,
                                           LENET_CONV1_M, LENET_CONV1_K),
        .kernel_h = LENET_CONV1_KH,
        .kernel_w = LENET_CONV1_KW,
        .pad_h = 2,
        .pad_w = 2,
        .stride_h = 1,
        .stride_w = 1,
        .out_h = LENET_CONV1_OUT_H,
        .out_w = LENET_CONV1_OUT_W,
    };
    lenet_oihw1_args = (GPGPUOihwToKoArgs) {
        .input = gpgpu_tensor_make_oihw_i32(lenet_conv1_weight_addr,
                                            LENET_CONV1_O, LENET_IN_C,
                                            LENET_CONV1_KH, LENET_CONV1_KW),
        .output = gpgpu_tensor_make_ko_i32(lenet_conv1_ko_addr,
                                           LENET_CONV1_K, LENET_CONV1_O),
        .out_channels = LENET_CONV1_O,
        .in_channels = LENET_IN_C,
        .kernel_h = LENET_CONV1_KH,
        .kernel_w = LENET_CONV1_KW,
    };
    lenet_conv1_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(lenet_conv1_col_addr,
                                      LENET_CONV1_M, LENET_CONV1_K),
        .b = gpgpu_tensor_make_ko_i32(lenet_conv1_ko_addr,
                                      LENET_CONV1_K, LENET_CONV1_O),
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_conv1_partial_addr,
            LENET_CONV1_M * LENET_CONV1_O * LENET_CONV1_K),
        .m = LENET_CONV1_M,
        .k = LENET_CONV1_K,
        .o = LENET_CONV1_O,
    };
    lenet_conv1_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_conv1_partial_addr,
            LENET_CONV1_M * LENET_CONV1_O * LENET_CONV1_K),
        .c = gpgpu_tensor_make_mo_i32(lenet_conv1_mo_addr,
                                      LENET_CONV1_M, LENET_CONV1_O),
        .bias = gpgpu_tensor_make_1d_i32(lenet_conv1_bias_addr,
                                         LENET_CONV1_O),
        .m = LENET_CONV1_M,
        .k = LENET_CONV1_K,
        .o = LENET_CONV1_O,
        .output_shift = SUNNYHAZE_LENET_Q8_SHIFT,
        .has_bias = 1,
    };
    lenet_mo1_args = (GPGPUMoToNchwArgs) {
        .input = gpgpu_tensor_make_mo_i32(lenet_conv1_mo_addr,
                                          LENET_CONV1_M, LENET_CONV1_O),
        .output = gpgpu_tensor_make_nchw_i32(
            lenet_conv1_nchw_addr, LENET_N, LENET_CONV1_O,
            LENET_CONV1_OUT_H, LENET_CONV1_OUT_W),
        .n = LENET_N,
        .c = LENET_CONV1_O,
        .h = LENET_CONV1_OUT_H,
        .w = LENET_CONV1_OUT_W,
    };
    lenet_relu1_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_nchw_i32(
            lenet_conv1_nchw_addr, LENET_N, LENET_CONV1_O,
            LENET_CONV1_OUT_H, LENET_CONV1_OUT_W),
        .output = gpgpu_tensor_make_nchw_i32(
            lenet_conv1_nchw_addr, LENET_N, LENET_CONV1_O,
            LENET_CONV1_OUT_H, LENET_CONV1_OUT_W),
    };
    lenet_pool1_args = (GPGPUMaxPool2DArgs) {
        .input = gpgpu_tensor_make_nchw_i32(
            lenet_conv1_nchw_addr, LENET_N, LENET_CONV1_O,
            LENET_CONV1_OUT_H, LENET_CONV1_OUT_W),
        .output = gpgpu_tensor_make_nchw_i32(
            lenet_pool1_addr, LENET_N, LENET_CONV1_O,
            LENET_POOL1_H, LENET_POOL1_W),
        .kernel_h = 2,
        .kernel_w = 2,
        .stride_h = 2,
        .stride_w = 2,
    };

    lenet_im2col2_args = (GPGPUIm2ColArgs) {
        .input = gpgpu_tensor_make_nchw_i32(lenet_pool1_addr, LENET_N,
                                            LENET_CONV1_O, LENET_POOL1_H,
                                            LENET_POOL1_W),
        .output = gpgpu_tensor_make_mk_i32(lenet_conv2_col_addr,
                                           LENET_CONV2_M, LENET_CONV2_K),
        .kernel_h = LENET_CONV2_KH,
        .kernel_w = LENET_CONV2_KW,
        .stride_h = 1,
        .stride_w = 1,
        .out_h = LENET_CONV2_OUT_H,
        .out_w = LENET_CONV2_OUT_W,
    };
    lenet_oihw2_args = (GPGPUOihwToKoArgs) {
        .input = gpgpu_tensor_make_oihw_i32(lenet_conv2_weight_addr,
                                            LENET_CONV2_O, LENET_CONV1_O,
                                            LENET_CONV2_KH, LENET_CONV2_KW),
        .output = gpgpu_tensor_make_ko_i32(lenet_conv2_ko_addr,
                                           LENET_CONV2_K, LENET_CONV2_O),
        .out_channels = LENET_CONV2_O,
        .in_channels = LENET_CONV1_O,
        .kernel_h = LENET_CONV2_KH,
        .kernel_w = LENET_CONV2_KW,
    };
    lenet_conv2_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(lenet_conv2_col_addr,
                                      LENET_CONV2_M, LENET_CONV2_K),
        .b = gpgpu_tensor_make_ko_i32(lenet_conv2_ko_addr,
                                      LENET_CONV2_K, LENET_CONV2_O),
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_conv2_partial_addr,
            LENET_CONV2_M * LENET_CONV2_O * LENET_CONV2_K),
        .m = LENET_CONV2_M,
        .k = LENET_CONV2_K,
        .o = LENET_CONV2_O,
    };
    lenet_conv2_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_conv2_partial_addr,
            LENET_CONV2_M * LENET_CONV2_O * LENET_CONV2_K),
        .c = gpgpu_tensor_make_mo_i32(lenet_conv2_mo_addr,
                                      LENET_CONV2_M, LENET_CONV2_O),
        .bias = gpgpu_tensor_make_1d_i32(lenet_conv2_bias_addr,
                                         LENET_CONV2_O),
        .m = LENET_CONV2_M,
        .k = LENET_CONV2_K,
        .o = LENET_CONV2_O,
        .output_shift = SUNNYHAZE_LENET_Q8_SHIFT,
        .has_bias = 1,
    };
    lenet_mo2_args = (GPGPUMoToNchwArgs) {
        .input = gpgpu_tensor_make_mo_i32(lenet_conv2_mo_addr,
                                          LENET_CONV2_M, LENET_CONV2_O),
        .output = gpgpu_tensor_make_nchw_i32(
            lenet_conv2_nchw_addr, LENET_N, LENET_CONV2_O,
            LENET_CONV2_OUT_H, LENET_CONV2_OUT_W),
        .n = LENET_N,
        .c = LENET_CONV2_O,
        .h = LENET_CONV2_OUT_H,
        .w = LENET_CONV2_OUT_W,
    };
    lenet_relu2_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_nchw_i32(
            lenet_conv2_nchw_addr, LENET_N, LENET_CONV2_O,
            LENET_CONV2_OUT_H, LENET_CONV2_OUT_W),
        .output = gpgpu_tensor_make_nchw_i32(
            lenet_conv2_nchw_addr, LENET_N, LENET_CONV2_O,
            LENET_CONV2_OUT_H, LENET_CONV2_OUT_W),
    };
    lenet_pool2_args = (GPGPUMaxPool2DArgs) {
        .input = gpgpu_tensor_make_nchw_i32(
            lenet_conv2_nchw_addr, LENET_N, LENET_CONV2_O,
            LENET_CONV2_OUT_H, LENET_CONV2_OUT_W),
        .output = gpgpu_tensor_make_nchw_i32(
            lenet_pool2_addr, LENET_N, LENET_CONV2_O,
            LENET_POOL2_H, LENET_POOL2_W),
        .kernel_h = 2,
        .kernel_w = 2,
        .stride_h = 2,
        .stride_w = 2,
    };

    lenet_fc1_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(lenet_pool2_addr, 1, LENET_FC1_IN),
        .b = gpgpu_tensor_make_ko_i32(lenet_fc1_weight_ko_addr,
                                      LENET_FC1_IN, LENET_FC1_OUT),
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_fc1_partial_addr, LENET_FC1_OUT * LENET_FC1_IN),
        .m = 1,
        .k = LENET_FC1_IN,
        .o = LENET_FC1_OUT,
    };
    lenet_fc1_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_fc1_partial_addr, LENET_FC1_OUT * LENET_FC1_IN),
        .c = gpgpu_tensor_make_mo_i32(lenet_fc1_out_addr, 1,
                                      LENET_FC1_OUT),
        .bias = gpgpu_tensor_make_1d_i32(lenet_fc1_bias_addr,
                                         LENET_FC1_OUT),
        .m = 1,
        .k = LENET_FC1_IN,
        .o = LENET_FC1_OUT,
        .output_shift = SUNNYHAZE_LENET_Q8_SHIFT,
        .has_bias = 1,
    };
    lenet_relu3_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_1d_i32(lenet_fc1_out_addr, LENET_FC1_OUT),
        .output = gpgpu_tensor_make_1d_i32(lenet_fc1_out_addr, LENET_FC1_OUT),
    };
    lenet_fc2_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(lenet_fc1_out_addr, 1, LENET_FC1_OUT),
        .b = gpgpu_tensor_make_ko_i32(lenet_fc2_weight_ko_addr,
                                      LENET_FC1_OUT, LENET_FC2_OUT),
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_fc2_partial_addr, LENET_FC2_OUT * LENET_FC1_OUT),
        .m = 1,
        .k = LENET_FC1_OUT,
        .o = LENET_FC2_OUT,
    };
    lenet_fc2_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_fc2_partial_addr, LENET_FC2_OUT * LENET_FC1_OUT),
        .c = gpgpu_tensor_make_mo_i32(lenet_fc2_out_addr, 1,
                                      LENET_FC2_OUT),
        .bias = gpgpu_tensor_make_1d_i32(lenet_fc2_bias_addr,
                                         LENET_FC2_OUT),
        .m = 1,
        .k = LENET_FC1_OUT,
        .o = LENET_FC2_OUT,
        .output_shift = SUNNYHAZE_LENET_Q8_SHIFT,
        .has_bias = 1,
    };
    lenet_relu4_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_1d_i32(lenet_fc2_out_addr, LENET_FC2_OUT),
        .output = gpgpu_tensor_make_1d_i32(lenet_fc2_out_addr, LENET_FC2_OUT),
    };
    lenet_fc3_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(lenet_fc2_out_addr, 1, LENET_FC2_OUT),
        .b = gpgpu_tensor_make_ko_i32(lenet_fc3_weight_ko_addr,
                                      LENET_FC2_OUT, LENET_FC3_OUT),
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_fc3_partial_addr, LENET_FC3_OUT * LENET_FC2_OUT),
        .m = 1,
        .k = LENET_FC2_OUT,
        .o = LENET_FC3_OUT,
    };
    lenet_fc3_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            lenet_fc3_partial_addr, LENET_FC3_OUT * LENET_FC2_OUT),
        .c = gpgpu_tensor_make_mo_i32(lenet_logits_addr, 1,
                                      LENET_FC3_OUT),
        .bias = gpgpu_tensor_make_1d_i32(lenet_fc3_bias_addr,
                                         LENET_FC3_OUT),
        .m = 1,
        .k = LENET_FC2_OUT,
        .o = LENET_FC3_OUT,
        .output_shift = SUNNYHAZE_LENET_Q8_SHIFT,
        .has_bias = 1,
    };

    ret = upload_args_checked(dev, &lenet_args_addr[0], &lenet_im2col1_args,
                              sizeof(lenet_im2col1_args),
                              "args(lenet_im2col1)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, im2col_kernel_addr,
             lenet_args_addr[0],
             (GPGPURuntimeDim3){ LENET_N, LENET_CONV1_OUT_H,
                                 LENET_CONV1_OUT_W },
             (GPGPURuntimeDim3){ LENET_CONV1_KW, LENET_CONV1_KH,
                                 LENET_IN_C });
    ret = upload_args_checked(dev, &lenet_args_addr[1], &lenet_oihw1_args,
                              sizeof(lenet_oihw1_args),
                              "args(lenet_oihw1)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, oihw_to_ko_kernel_addr,
             lenet_args_addr[1],
             (GPGPURuntimeDim3){ LENET_CONV1_O, LENET_IN_C,
                                 LENET_CONV1_KH },
             (GPGPURuntimeDim3){ LENET_CONV1_KW, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[2],
                              &lenet_conv1_partial_args,
                              sizeof(lenet_conv1_partial_args),
                              "args(lenet_conv1_partial)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_partial_kernel_addr,
             lenet_args_addr[2],
             (GPGPURuntimeDim3){ LENET_CONV1_M, LENET_CONV1_O, 1 },
             (GPGPURuntimeDim3){ LENET_CONV1_K, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[3],
                              &lenet_conv1_reduce_args,
                              sizeof(lenet_conv1_reduce_args),
                              "args(lenet_conv1_reduce)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_reduce_kernel_addr,
             lenet_args_addr[3],
             (GPGPURuntimeDim3){ LENET_CONV1_M, 1, 1 },
             (GPGPURuntimeDim3){ LENET_CONV1_O, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[4], &lenet_mo1_args,
                              sizeof(lenet_mo1_args), "args(lenet_mo1)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, layout_kernel_addr,
             lenet_args_addr[4],
             (GPGPURuntimeDim3){ LENET_N, LENET_CONV1_O,
                                 LENET_CONV1_OUT_H },
             (GPGPURuntimeDim3){ LENET_CONV1_OUT_W, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[5], &lenet_relu1_args,
                              sizeof(lenet_relu1_args), "args(lenet_relu1)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, relu_kernel_addr,
             lenet_args_addr[5],
             (GPGPURuntimeDim3){ 147, 1, 1 },
             (GPGPURuntimeDim3){ 32, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[6], &lenet_pool1_args,
                              sizeof(lenet_pool1_args), "args(lenet_pool1)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, pool_kernel_addr,
             lenet_args_addr[6],
             (GPGPURuntimeDim3){ LENET_N, LENET_CONV1_O, LENET_POOL1_H },
             (GPGPURuntimeDim3){ LENET_POOL1_W, 1, 1 });

    ret = upload_args_checked(dev, &lenet_args_addr[7], &lenet_im2col2_args,
                              sizeof(lenet_im2col2_args),
                              "args(lenet_im2col2)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, im2col_kernel_addr,
             lenet_args_addr[7],
             (GPGPURuntimeDim3){ LENET_N, LENET_CONV2_OUT_H,
                                 LENET_CONV2_OUT_W },
             (GPGPURuntimeDim3){ LENET_CONV2_KW, LENET_CONV2_KH,
                                 LENET_CONV1_O });
    ret = upload_args_checked(dev, &lenet_args_addr[8], &lenet_oihw2_args,
                              sizeof(lenet_oihw2_args),
                              "args(lenet_oihw2)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, oihw_to_ko_kernel_addr,
             lenet_args_addr[8],
             (GPGPURuntimeDim3){ LENET_CONV2_O, LENET_CONV1_O,
                                 LENET_CONV2_KH },
             (GPGPURuntimeDim3){ LENET_CONV2_KW, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[9],
                              &lenet_conv2_partial_args,
                              sizeof(lenet_conv2_partial_args),
                              "args(lenet_conv2_partial)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_partial_kernel_addr,
             lenet_args_addr[9],
             (GPGPURuntimeDim3){ LENET_CONV2_M, LENET_CONV2_O, 1 },
             (GPGPURuntimeDim3){ LENET_CONV2_K, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[10],
                              &lenet_conv2_reduce_args,
                              sizeof(lenet_conv2_reduce_args),
                              "args(lenet_conv2_reduce)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_reduce_kernel_addr,
             lenet_args_addr[10],
             (GPGPURuntimeDim3){ LENET_CONV2_M, 1, 1 },
             (GPGPURuntimeDim3){ LENET_CONV2_O, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[11], &lenet_mo2_args,
                              sizeof(lenet_mo2_args), "args(lenet_mo2)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, layout_kernel_addr,
             lenet_args_addr[11],
             (GPGPURuntimeDim3){ LENET_N, LENET_CONV2_O,
                                 LENET_CONV2_OUT_H },
             (GPGPURuntimeDim3){ LENET_CONV2_OUT_W, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[12], &lenet_relu2_args,
                              sizeof(lenet_relu2_args), "args(lenet_relu2)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, relu_kernel_addr,
             lenet_args_addr[12],
             (GPGPURuntimeDim3){ 50, 1, 1 },
             (GPGPURuntimeDim3){ 32, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[13], &lenet_pool2_args,
                              sizeof(lenet_pool2_args), "args(lenet_pool2)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, pool_kernel_addr,
             lenet_args_addr[13],
             (GPGPURuntimeDim3){ LENET_N, LENET_CONV2_O, LENET_POOL2_H },
             (GPGPURuntimeDim3){ LENET_POOL2_W, 1, 1 });

    ret = upload_args_checked(dev, &lenet_args_addr[14],
                              &lenet_fc1_partial_args,
                              sizeof(lenet_fc1_partial_args),
                              "args(lenet_fc1_partial)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_partial_kernel_addr,
             lenet_args_addr[14],
             (GPGPURuntimeDim3){ 1, LENET_FC1_OUT, 1 },
             (GPGPURuntimeDim3){ LENET_FC1_IN, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[15],
                              &lenet_fc1_reduce_args,
                              sizeof(lenet_fc1_reduce_args),
                              "args(lenet_fc1_reduce)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_reduce_kernel_addr,
             lenet_args_addr[15],
             (GPGPURuntimeDim3){ 1, 1, 1 },
             (GPGPURuntimeDim3){ LENET_FC1_OUT, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[16], &lenet_relu3_args,
                              sizeof(lenet_relu3_args), "args(lenet_relu3)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, relu_kernel_addr,
             lenet_args_addr[16],
             (GPGPURuntimeDim3){ 4, 1, 1 },
             (GPGPURuntimeDim3){ 32, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[17],
                              &lenet_fc2_partial_args,
                              sizeof(lenet_fc2_partial_args),
                              "args(lenet_fc2_partial)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_partial_kernel_addr,
             lenet_args_addr[17],
             (GPGPURuntimeDim3){ 1, LENET_FC2_OUT, 1 },
             (GPGPURuntimeDim3){ LENET_FC1_OUT, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[18],
                              &lenet_fc2_reduce_args,
                              sizeof(lenet_fc2_reduce_args),
                              "args(lenet_fc2_reduce)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_reduce_kernel_addr,
             lenet_args_addr[18],
             (GPGPURuntimeDim3){ 1, 1, 1 },
             (GPGPURuntimeDim3){ LENET_FC2_OUT, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[19], &lenet_relu4_args,
                              sizeof(lenet_relu4_args), "args(lenet_relu4)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, relu_kernel_addr,
             lenet_args_addr[19],
             (GPGPURuntimeDim3){ 3, 1, 1 },
             (GPGPURuntimeDim3){ 32, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[20],
                              &lenet_fc3_partial_args,
                              sizeof(lenet_fc3_partial_args),
                              "args(lenet_fc3_partial)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_partial_kernel_addr,
             lenet_args_addr[20],
             (GPGPURuntimeDim3){ 1, LENET_FC3_OUT, 1 },
             (GPGPURuntimeDim3){ LENET_FC2_OUT, 1, 1 });
    ret = upload_args_checked(dev, &lenet_args_addr[21],
                              &lenet_fc3_reduce_args,
                              sizeof(lenet_fc3_reduce_args),
                              "args(lenet_fc3_reduce)");
    if (ret < 0) return ret;
    add_node(lenet_nodes, &lenet_node_count, matmul_reduce_kernel_addr,
             lenet_args_addr[21],
             (GPGPURuntimeDim3){ 1, 1, 1 },
             (GPGPURuntimeDim3){ LENET_FC3_OUT, 1, 1 });

    for (uint32_t sample = 0; sample < MNIST_SAMPLE_COUNT; ++sample) {
        ret = gpgpu_write(dev, lenet_input_addr, mnist_samples_q8[sample],
                          LENET_N * LENET_IN_C * LENET_IN_H * LENET_IN_W *
                          sizeof(int32_t));
        if (ret < 0) {
            report_ret("gpgpu_write(lenet_input)", ret);
            return ret;
        }

        ret = run_nodes(dev, lenet_nodes, lenet_node_count, "gpgpu lenet");
        if (ret < 0) {
            return ret;
        }
        ret = gpgpu_read(dev, lenet_logits_addr, lenet_logits,
                         sizeof(lenet_logits));
        if (ret < 0) {
            report_ret("gpgpu_read(lenet_logits)", ret);
            return ret;
        }

        lenet_pred = 0;
        uart_puts("gpgpu lenet_sample=");
        uart_puthex32(sample);
        uart_puts("\n");
        for (uint32_t i = 0; i < LENET_FC3_OUT; ++i) {
            uart_puts("gpgpu lenet_logit[");
            uart_puthex32(i);
            uart_puts("]=");
            uart_puthex32((uint32_t)lenet_logits[i]);
            uart_puts("\n");
            if (lenet_logits[i] > lenet_logits[lenet_pred]) {
                lenet_pred = i;
            }
        }
        trace_u32("gpgpu lenet_expected", mnist_sample_labels[sample]);
        trace_u32("gpgpu lenet_pred", lenet_pred);
        uart_puts("gpgpu lenet_sample ");
        if (lenet_pred == mnist_sample_labels[sample]) {
            uart_puts("PASS\n");
            ++lenet_correct;
        } else {
            uart_puts("FAIL\n");
        }
    }

    trace_u32("gpgpu lenet_correct", lenet_correct);
    trace_u32("gpgpu lenet_total", MNIST_SAMPLE_COUNT);
    uart_puts("gpgpu lenet DIAGNOSTIC\n");
    return 0;
}
