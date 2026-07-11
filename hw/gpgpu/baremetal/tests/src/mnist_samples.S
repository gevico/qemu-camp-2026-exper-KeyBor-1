    .section .rodata
    .balign 4
    .global mnist_samples_q8_blob_start
mnist_samples_q8_blob_start:
    .incbin "../assets/lenet/mnist_samples_q8.bin"
    .balign 4
    .global mnist_samples_q8_blob_end
mnist_samples_q8_blob_end:
