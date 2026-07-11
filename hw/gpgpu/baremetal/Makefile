CROSS_COMPILE ?= riscv64-unknown-elf-
CC := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

BUILD_DIR := build
SRC_DIR := src
TEST_DIR := tests
TEST_SRC_DIR := $(TEST_DIR)/src
TEST_INCLUDE_DIR := $(TEST_DIR)/include
INCLUDE_DIR := include
SCRIPT_DIR := scripts
KERNEL_DIR := $(SRC_DIR)/kernels
GPGPU_KERNEL_DIR := ../kernels
DRIVER_DIR := drivers
RUNTIME_DIR := ../runtime
TARGET := $(BUILD_DIR)/gpgpu-smoke.elf
BIN := $(BUILD_DIR)/gpgpu-smoke.bin
UART_TARGET := $(BUILD_DIR)/uart-smoke.elf
UART_BIN := $(BUILD_DIR)/uart-smoke.bin
THREAD_ADD_KERNEL_INC := $(BUILD_DIR)/thread_add_kernel.inc
STACK_SMOKE_KERNEL_INC := $(BUILD_DIR)/stack_smoke_kernel.inc
RELU_I32_KERNEL_INC := $(BUILD_DIR)/relu_i32_kernel.inc
LINEAR_I32_KERNEL_INC := $(BUILD_DIR)/linear_i32_kernel.inc
LINEAR_PARTIAL_I32_KERNEL_INC := $(BUILD_DIR)/linear_partial_i32_kernel.inc
LINEAR_REDUCE_I32_KERNEL_INC := $(BUILD_DIR)/linear_reduce_i32_kernel.inc
MATMUL_PARTIAL_I32_KERNEL_INC := $(BUILD_DIR)/matmul_partial_i32_kernel.inc
MATMUL_REDUCE_I32_KERNEL_INC := $(BUILD_DIR)/matmul_reduce_i32_kernel.inc
IM2COL_I32_KERNEL_INC := $(BUILD_DIR)/im2col_i32_kernel.inc
OIHW_TO_KO_I32_KERNEL_INC := $(BUILD_DIR)/oihw_to_ko_i32_kernel.inc
MO_TO_NCHW_I32_KERNEL_INC := $(BUILD_DIR)/mo_to_nchw_i32_kernel.inc
MAXPOOL_I32_KERNEL_INC := $(BUILD_DIR)/maxpool_i32_kernel.inc
KERNEL_INCS := \
	$(THREAD_ADD_KERNEL_INC) \
	$(STACK_SMOKE_KERNEL_INC) \
	$(RELU_I32_KERNEL_INC) \
	$(LINEAR_I32_KERNEL_INC) \
	$(LINEAR_PARTIAL_I32_KERNEL_INC) \
	$(LINEAR_REDUCE_I32_KERNEL_INC) \
	$(MATMUL_PARTIAL_I32_KERNEL_INC) \
	$(MATMUL_REDUCE_I32_KERNEL_INC) \
	$(IM2COL_I32_KERNEL_INC) \
	$(OIHW_TO_KO_I32_KERNEL_INC) \
	$(MO_TO_NCHW_I32_KERNEL_INC) \
	$(MAXPOOL_I32_KERNEL_INC)
TEST_SRCS := \
	test_common.c \
	smoke.c \
	basic_smoke.c \
	thread_add_smoke.c \
	stack_smoke.c \
	relu_smoke.c \
	linear_smoke.c \
	matmul_smoke.c \
	qmatmul_smoke.c \
	layout_smoke.c \
	conv_smoke.c \
	maxpool_smoke.c \
	lenet_smoke.c
TEST_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

CFLAGS := -march=rv64imac -mabi=lp64 -mcmodel=medany
CFLAGS += -Wall -Wextra -ffreestanding -fno-builtin -fno-common
CFLAGS += -O2 -I. -I$(INCLUDE_DIR) -I$(DRIVER_DIR)/include
CFLAGS += -I$(TEST_INCLUDE_DIR) -I$(RUNTIME_DIR)/include
KERNEL_CFLAGS := -march=rv32im -mabi=ilp32 -Wall -Wextra
KERNEL_CFLAGS += -ffreestanding -fno-builtin -fno-common -fno-pic
KERNEL_CFLAGS += -fno-asynchronous-unwind-tables -fno-unwind-tables -Os
KERNEL_CFLAGS += -I$(RUNTIME_DIR)/include
LDFLAGS := -nostdlib -T $(SCRIPT_DIR)/linker.ld

OBJS := \
	$(BUILD_DIR)/start.o \
	$(BUILD_DIR)/main.o \
	$(TEST_OBJS) \
	$(BUILD_DIR)/gpgpu_runtime.o \
	$(BUILD_DIR)/gpgpu_pci.o

.PHONY: all clean run uart run-uart
all: $(TARGET) $(BIN)
uart: $(UART_TARGET) $(UART_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/start.o: $(SCRIPT_DIR)/start.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_SRC_DIR)/%.c $(KERNEL_INCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/thread_add_kernel.o: $(GPGPU_KERNEL_DIR)/thread_add_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/thread_add_kernel.bin: $(BUILD_DIR)/thread_add_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/thread_add_kernel.dis: $(BUILD_DIR)/thread_add_kernel.o
	$(OBJDUMP) -dr $< > $@

$(THREAD_ADD_KERNEL_INC): $(BUILD_DIR)/thread_add_kernel.bin $(BUILD_DIR)/thread_add_kernel.dis
	printf 'static const uint32_t thread_add_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/thread_add_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/stack_smoke_kernel.o: $(GPGPU_KERNEL_DIR)/stack_smoke_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/stack_smoke_kernel.bin: $(BUILD_DIR)/stack_smoke_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/stack_smoke_kernel.dis: $(BUILD_DIR)/stack_smoke_kernel.o
	$(OBJDUMP) -dr $< > $@

$(STACK_SMOKE_KERNEL_INC): $(BUILD_DIR)/stack_smoke_kernel.bin $(BUILD_DIR)/stack_smoke_kernel.dis
	printf 'static const uint32_t stack_smoke_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/stack_smoke_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/relu_i32_kernel.o: $(GPGPU_KERNEL_DIR)/relu_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/relu_i32_kernel.bin: $(BUILD_DIR)/relu_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/relu_i32_kernel.dis: $(BUILD_DIR)/relu_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(RELU_I32_KERNEL_INC): $(BUILD_DIR)/relu_i32_kernel.bin $(BUILD_DIR)/relu_i32_kernel.dis
	printf 'static const uint32_t relu_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/relu_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/linear_i32_kernel.o: $(GPGPU_KERNEL_DIR)/linear_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_i32_kernel.bin: $(BUILD_DIR)/linear_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/linear_i32_kernel.dis: $(BUILD_DIR)/linear_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(LINEAR_I32_KERNEL_INC): $(BUILD_DIR)/linear_i32_kernel.bin $(BUILD_DIR)/linear_i32_kernel.dis
	printf 'static const uint32_t linear_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/linear_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/linear_partial_i32_kernel.o: $(GPGPU_KERNEL_DIR)/linear_partial_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_partial_i32_kernel.bin: $(BUILD_DIR)/linear_partial_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/linear_partial_i32_kernel.dis: $(BUILD_DIR)/linear_partial_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(LINEAR_PARTIAL_I32_KERNEL_INC): $(BUILD_DIR)/linear_partial_i32_kernel.bin $(BUILD_DIR)/linear_partial_i32_kernel.dis
	printf 'static const uint32_t linear_partial_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/linear_partial_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/linear_reduce_i32_kernel.o: $(GPGPU_KERNEL_DIR)/linear_reduce_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_reduce_i32_kernel.bin: $(BUILD_DIR)/linear_reduce_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/linear_reduce_i32_kernel.dis: $(BUILD_DIR)/linear_reduce_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(LINEAR_REDUCE_I32_KERNEL_INC): $(BUILD_DIR)/linear_reduce_i32_kernel.bin $(BUILD_DIR)/linear_reduce_i32_kernel.dis
	printf 'static const uint32_t linear_reduce_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/linear_reduce_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/matmul_partial_i32_kernel.o: $(GPGPU_KERNEL_DIR)/matmul_partial_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/matmul_partial_i32_kernel.bin: $(BUILD_DIR)/matmul_partial_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/matmul_partial_i32_kernel.dis: $(BUILD_DIR)/matmul_partial_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(MATMUL_PARTIAL_I32_KERNEL_INC): $(BUILD_DIR)/matmul_partial_i32_kernel.bin $(BUILD_DIR)/matmul_partial_i32_kernel.dis
	printf 'static const uint32_t matmul_partial_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/matmul_partial_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/matmul_reduce_i32_kernel.o: $(GPGPU_KERNEL_DIR)/matmul_reduce_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/matmul_reduce_i32_kernel.bin: $(BUILD_DIR)/matmul_reduce_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/matmul_reduce_i32_kernel.dis: $(BUILD_DIR)/matmul_reduce_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(MATMUL_REDUCE_I32_KERNEL_INC): $(BUILD_DIR)/matmul_reduce_i32_kernel.bin $(BUILD_DIR)/matmul_reduce_i32_kernel.dis
	printf 'static const uint32_t matmul_reduce_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/matmul_reduce_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/im2col_i32_kernel.o: $(GPGPU_KERNEL_DIR)/im2col_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/im2col_i32_kernel.bin: $(BUILD_DIR)/im2col_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/im2col_i32_kernel.dis: $(BUILD_DIR)/im2col_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(IM2COL_I32_KERNEL_INC): $(BUILD_DIR)/im2col_i32_kernel.bin $(BUILD_DIR)/im2col_i32_kernel.dis
	printf 'static const uint32_t im2col_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/im2col_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/oihw_to_ko_i32_kernel.o: $(GPGPU_KERNEL_DIR)/oihw_to_ko_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/oihw_to_ko_i32_kernel.bin: $(BUILD_DIR)/oihw_to_ko_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/oihw_to_ko_i32_kernel.dis: $(BUILD_DIR)/oihw_to_ko_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(OIHW_TO_KO_I32_KERNEL_INC): $(BUILD_DIR)/oihw_to_ko_i32_kernel.bin $(BUILD_DIR)/oihw_to_ko_i32_kernel.dis
	printf 'static const uint32_t oihw_to_ko_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/oihw_to_ko_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/mo_to_nchw_i32_kernel.o: $(GPGPU_KERNEL_DIR)/mo_to_nchw_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/mo_to_nchw_i32_kernel.bin: $(BUILD_DIR)/mo_to_nchw_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/mo_to_nchw_i32_kernel.dis: $(BUILD_DIR)/mo_to_nchw_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(MO_TO_NCHW_I32_KERNEL_INC): $(BUILD_DIR)/mo_to_nchw_i32_kernel.bin $(BUILD_DIR)/mo_to_nchw_i32_kernel.dis
	printf 'static const uint32_t mo_to_nchw_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/mo_to_nchw_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/maxpool_i32_kernel.o: $(GPGPU_KERNEL_DIR)/maxpool_i32_kernel.c | $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/maxpool_i32_kernel.bin: $(BUILD_DIR)/maxpool_i32_kernel.o
	$(OBJCOPY) -O binary -j .text $< $@

$(BUILD_DIR)/maxpool_i32_kernel.dis: $(BUILD_DIR)/maxpool_i32_kernel.o
	$(OBJDUMP) -dr $< > $@

$(MAXPOOL_I32_KERNEL_INC): $(BUILD_DIR)/maxpool_i32_kernel.bin $(BUILD_DIR)/maxpool_i32_kernel.dis
	printf 'static const uint32_t maxpool_i32_kernel_code[] = {\n' > $@
	od -An -v -tx4 $(BUILD_DIR)/maxpool_i32_kernel.bin | \
	    awk '{ for (i = 1; i <= NF; ++i) printf "    0x%s,\n", $$i }' >> $@
	printf '};\n' >> $@

$(BUILD_DIR)/uart_smoke.o: $(SRC_DIR)/uart_smoke.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gpgpu_runtime.o: $(RUNTIME_DIR)/src/gpgpu_runtime.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gpgpu_pci.o: $(DRIVER_DIR)/src/gpgpu_pci.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(SCRIPT_DIR)/linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@

$(BIN): $(TARGET)
	$(OBJCOPY) -O binary $< $@

$(UART_TARGET): $(BUILD_DIR)/start.o $(BUILD_DIR)/uart_smoke.o $(SCRIPT_DIR)/linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(BUILD_DIR)/start.o $(BUILD_DIR)/uart_smoke.o -o $@

$(UART_BIN): $(UART_TARGET)
	$(OBJCOPY) -O binary $< $@

run: $(TARGET)
	qemu-system-riscv64 -machine virt -m 256M -nographic -bios none \
	    -device gpgpu -kernel $(TARGET)

run-uart: $(UART_TARGET)
	qemu-system-riscv64 -machine virt -m 256M -nographic -bios none \
	    -kernel $(UART_TARGET)

clean:
	rm -rf $(BUILD_DIR)
