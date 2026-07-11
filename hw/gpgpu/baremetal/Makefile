CROSS_COMPILE ?= riscv64-unknown-elf-
CC := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

BUILD_DIR := build
SRC_DIR := src
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

CFLAGS := -march=rv64imac -mabi=lp64 -mcmodel=medany
CFLAGS += -Wall -Wextra -ffreestanding -fno-builtin -fno-common
CFLAGS += -O2 -I. -I$(INCLUDE_DIR) -I$(DRIVER_DIR)/include
CFLAGS += -I$(RUNTIME_DIR)/include
KERNEL_CFLAGS := -march=rv32im -mabi=ilp32 -Wall -Wextra
KERNEL_CFLAGS += -ffreestanding -fno-builtin -fno-common -fno-pic
KERNEL_CFLAGS += -fno-asynchronous-unwind-tables -fno-unwind-tables -Os
LDFLAGS := -nostdlib -T $(SCRIPT_DIR)/linker.ld

OBJS := \
	$(BUILD_DIR)/start.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/gpgpu_runtime.o \
	$(BUILD_DIR)/gpgpu_pci.o

.PHONY: all clean run uart run-uart
all: $(TARGET) $(BIN)
uart: $(UART_TARGET) $(UART_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/start.o: $(SCRIPT_DIR)/start.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c $(THREAD_ADD_KERNEL_INC) | $(BUILD_DIR)
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
