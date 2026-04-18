#ifndef G233_SPI_H
#define G233_SPI_H

#include <stdbool.h>
#include <stdint.h>

#include "hw/core/sysbus.h"
#include "qom/object.h"

#define TYPE_G233_SPI "g233-spi"
typedef struct G233SpiState G233SpiState;

#define G233_SPI(obj) \
    OBJECT_CHECK(G233SpiState, (obj), TYPE_G233_SPI)

#define G233_SPI_MMIO_SIZE 0xFFF

#define G233_SPI_CR1_REG 0x00
#define G233_SPI_CR2_REG 0x04
#define G233_SPI_SR_REG  0x08
#define G233_SPI_DR_REG  0x0C

#define G233_SPI_CR1_SPE    (1u << 0)
#define G233_SPI_CR1_MSTR   (1u << 2)
#define G233_SPI_CR1_ERRIE  (1u << 5)
#define G233_SPI_CR1_RXNEIE (1u << 6)
#define G233_SPI_CR1_TXEIE  (1u << 7)
#define G233_SPI_CR1_RW_MASK \
    (G233_SPI_CR1_SPE | G233_SPI_CR1_MSTR | G233_SPI_CR1_ERRIE | \
     G233_SPI_CR1_RXNEIE | G233_SPI_CR1_TXEIE)

#define G233_SPI_CR2_CS_MASK 0x3u

#define G233_SPI_SR_RXNE    (1u << 0)
#define G233_SPI_SR_TXE     (1u << 1)
#define G233_SPI_SR_OVERRUN (1u << 4)
#define G233_SPI_SR_RW1C_MASK G233_SPI_SR_OVERRUN

#define G233_FLASH_CMD_WRITE_ENABLE 0x06
#define G233_FLASH_CMD_READ_STATUS  0x05
#define G233_FLASH_CMD_READ_DATA    0x03
#define G233_FLASH_CMD_PAGE_PROGRAM 0x02
#define G233_FLASH_CMD_SECTOR_ERASE 0x20
#define G233_FLASH_CMD_JEDEC_ID     0x9F

#define G233_FLASH_SR_BUSY 0x01
#define G233_FLASH_SECTOR_SIZE 4096u

typedef struct G233SpiFlashState {
    uint8_t *storage;
    uint32_t size;
    uint8_t jedec[3];
    bool write_enable;
    bool busy;
} G233SpiFlashState;

typedef enum G233SpiPhase {
    G233_SPI_PHASE_CMD = 0,
    G233_SPI_PHASE_ADDR,
    G233_SPI_PHASE_READ_STATUS,
    G233_SPI_PHASE_READ_JEDEC,
    G233_SPI_PHASE_READ_DATA,
    G233_SPI_PHASE_PROGRAM_DATA,
    G233_SPI_PHASE_IGNORE,
} G233SpiPhase;

struct G233SpiState {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;

    uint32_t cr1;
    uint32_t cr2;
    uint32_t sr;
    uint8_t dr;

    G233SpiFlashState flashes[2];

    int active_cs;
    uint8_t current_cmd;
    G233SpiPhase phase;
    uint32_t addr;
    int addr_bytes;
    int response_index;
};

#endif