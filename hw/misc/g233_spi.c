#include "qemu/osdep.h"
#include "hw/core/irq.h"
#include "hw/core/qdev.h"
#include "hw/core/sysbus.h"
#include "qom/object.h"
#include "system/memory.h"
#include "qemu/log.h"
#include "hw/misc/g233_spi.h"

#define G233_SPI_FLASH0_SIZE (2 * 1024 * 1024u)
#define G233_SPI_FLASH1_SIZE (4 * 1024 * 1024u)

static void g233_spi_update_irq(G233SpiState *s)
{
    bool enabled = s->cr1 & G233_SPI_CR1_SPE;
    bool level = false;

    if (enabled) {
        level = ((s->sr & G233_SPI_SR_TXE) && (s->cr1 & G233_SPI_CR1_TXEIE)) ||
                ((s->sr & G233_SPI_SR_RXNE) && (s->cr1 & G233_SPI_CR1_RXNEIE)) ||
                ((s->sr & G233_SPI_SR_OVERRUN) && (s->cr1 & G233_SPI_CR1_ERRIE));
    }

    qemu_set_irq(s->irq, level);
}

static void g233_spi_reset_session(G233SpiState *s)
{
    s->current_cmd = 0;
    s->phase = G233_SPI_PHASE_CMD;
    s->addr = 0;
    s->addr_bytes = 0;
    s->response_index = 0;
}

static G233SpiFlashState *g233_spi_current_flash(G233SpiState *s)
{
    if (s->active_cs < 0 || s->active_cs >= 2) {
        return NULL;
    }

    return &s->flashes[s->active_cs];
}

static uint8_t g233_spi_flash_status(const G233SpiFlashState *flash)
{
    return flash->busy ? G233_FLASH_SR_BUSY : 0;
}

static void g233_spi_erase_sector(G233SpiFlashState *flash, uint32_t addr)
{
    uint32_t sector = (addr / G233_FLASH_SECTOR_SIZE) * G233_FLASH_SECTOR_SIZE;

    for (uint32_t i = 0; i < G233_FLASH_SECTOR_SIZE; i++) {
        flash->storage[(sector + i) % flash->size] = 0xFF;
    }

    flash->write_enable = false;
}

static uint8_t g233_spi_flash_transfer(G233SpiState *s, uint8_t tx)
{
    G233SpiFlashState *flash = g233_spi_current_flash(s);

    if (!flash) {
        return 0xFF;
    }

    switch (s->phase) {
    case G233_SPI_PHASE_CMD:
        s->current_cmd = tx;
        s->addr = 0;
        s->addr_bytes = 0;
        s->response_index = 0;

        switch (tx) {
        case G233_FLASH_CMD_WRITE_ENABLE:
            flash->write_enable = true;
            s->phase = G233_SPI_PHASE_IGNORE;
            break;
        case G233_FLASH_CMD_READ_STATUS:
            s->phase = G233_SPI_PHASE_READ_STATUS;
            break;
        case G233_FLASH_CMD_JEDEC_ID:
            s->phase = G233_SPI_PHASE_READ_JEDEC;
            break;
        case G233_FLASH_CMD_READ_DATA:
        case G233_FLASH_CMD_SECTOR_ERASE:
        case G233_FLASH_CMD_PAGE_PROGRAM:
            s->phase = G233_SPI_PHASE_ADDR;
            break;
        default:
            s->phase = G233_SPI_PHASE_IGNORE;
            break;
        }
        return 0xFF;

    case G233_SPI_PHASE_ADDR:
        s->addr = (s->addr << 8) | tx;
        s->addr_bytes++;

        if (s->addr_bytes < 3) {
            return 0xFF;
        }

        s->addr %= flash->size;
        if (s->current_cmd == G233_FLASH_CMD_READ_DATA) {
            s->phase = G233_SPI_PHASE_READ_DATA;
        } else if (s->current_cmd == G233_FLASH_CMD_SECTOR_ERASE) {
            if (flash->write_enable) {
                g233_spi_erase_sector(flash, s->addr);
            }
            s->phase = G233_SPI_PHASE_IGNORE;
        } else if (s->current_cmd == G233_FLASH_CMD_PAGE_PROGRAM) {
            s->phase = G233_SPI_PHASE_PROGRAM_DATA;
        }
        return 0xFF;

    case G233_SPI_PHASE_READ_STATUS:
        return g233_spi_flash_status(flash);

    case G233_SPI_PHASE_READ_JEDEC:
        if (s->response_index < 3) {
            return flash->jedec[s->response_index++];
        }
        return 0xFF;

    case G233_SPI_PHASE_READ_DATA: {
        uint8_t value = flash->storage[s->addr];
        s->addr = (s->addr + 1) % flash->size;
        return value;
    }

    case G233_SPI_PHASE_PROGRAM_DATA:
        if (flash->write_enable) {
            flash->storage[s->addr] &= tx;
            s->addr = (s->addr + 1) % flash->size;
        }
        return 0xFF;

    case G233_SPI_PHASE_IGNORE:
    default:
        return 0xFF;
    }
}

static void g233_spi_select_cs(G233SpiState *s, uint32_t cs)
{
    int new_cs = cs & G233_SPI_CR2_CS_MASK;

    if (new_cs != s->active_cs) {
        g233_spi_reset_session(s);
        s->active_cs = new_cs;
    }

    s->cr2 = new_cs;
}

static void g233_spi_reset(DeviceState *dev)
{
    G233SpiState *s = G233_SPI(dev);

    s->cr1 = 0;
    s->cr2 = 0;
    s->sr = G233_SPI_SR_TXE;
    s->dr = 0;
    s->active_cs = 0;
    g233_spi_reset_session(s);

    for (int i = 0; i < 2; i++) {
        memset(s->flashes[i].storage, 0xFF, s->flashes[i].size);
        s->flashes[i].write_enable = false;
        s->flashes[i].busy = false;
    }

    g233_spi_update_irq(s);
}

static void g233_spi_realize(DeviceState *dev, Error **errp)
{
    G233SpiState *s = G233_SPI(dev);

    s->flashes[0].size = G233_SPI_FLASH0_SIZE;
    s->flashes[0].storage = g_malloc0(s->flashes[0].size);
    s->flashes[0].jedec[0] = 0xEF;
    s->flashes[0].jedec[1] = 0x30;
    s->flashes[0].jedec[2] = 0x15;

    s->flashes[1].size = G233_SPI_FLASH1_SIZE;
    s->flashes[1].storage = g_malloc0(s->flashes[1].size);
    s->flashes[1].jedec[0] = 0xEF;
    s->flashes[1].jedec[1] = 0x30;
    s->flashes[1].jedec[2] = 0x16;

    g233_spi_reset(dev);
}

static void g233_spi_unrealize(DeviceState *dev)
{
    G233SpiState *s = G233_SPI(dev);

    for (int i = 0; i < 2; i++) {
        g_free(s->flashes[i].storage);
        s->flashes[i].storage = NULL;
    }
}

static void g233_spi_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = g233_spi_realize;
    dc->unrealize = g233_spi_unrealize;
    device_class_set_legacy_reset(dc, g233_spi_reset);
}

static uint64_t g233_spi_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    G233SpiState *s = G233_SPI(opaque);
    uint32_t value;

    if (size != 4 || (addr & 0x3)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-spi: invalid read size/alignment addr=0x%" HWADDR_PRIx
                      " size=%u\n",
                      addr, size);
        return 0;
    }

    switch (addr) {
    case G233_SPI_CR1_REG:
        value = s->cr1;
        break;
    case G233_SPI_CR2_REG:
        value = s->cr2;
        break;
    case G233_SPI_SR_REG:
        value = s->sr;
        break;
    case G233_SPI_DR_REG:
        value = s->dr;
        s->sr &= ~G233_SPI_SR_RXNE;
        g233_spi_update_irq(s);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-spi: invalid read addr=0x%" HWADDR_PRIx "\n",
                      addr);
        value = 0;
        break;
    }

    return value;
}

static void g233_spi_mmio_write(void *opaque, hwaddr addr,
                                uint64_t data, unsigned size)
{
    G233SpiState *s = G233_SPI(opaque);
    uint32_t value = (uint32_t)data;

    if (size != 4 || (addr & 0x3)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-spi: invalid write size/alignment addr=0x%" HWADDR_PRIx
                      " size=%u data=0x%" PRIx64 "\n",
                      addr, size, data);
        return;
    }

    switch (addr) {
    case G233_SPI_CR1_REG:
        s->cr1 = value & G233_SPI_CR1_RW_MASK;
        break;
    case G233_SPI_CR2_REG:
        g233_spi_select_cs(s, value);
        break;
    case G233_SPI_SR_REG:
        s->sr &= ~(value & G233_SPI_SR_RW1C_MASK);
        break;
    case G233_SPI_DR_REG:
        if (!(s->cr1 & G233_SPI_CR1_SPE)) {
            break;
        }

        if (s->sr & G233_SPI_SR_RXNE) {
            s->sr |= G233_SPI_SR_OVERRUN;
        }

        s->dr = g233_spi_flash_transfer(s, value & 0xFF);
        s->sr |= G233_SPI_SR_RXNE | G233_SPI_SR_TXE;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-spi: invalid write addr=0x%" HWADDR_PRIx
                      " data=0x%" PRIx64 "\n",
                      addr, data);
        return;
    }

    g233_spi_update_irq(s);
}

static const MemoryRegionOps g233_spi_ops = {
    .read = g233_spi_mmio_read,
    .write = g233_spi_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl.min_access_size = 4,
    .impl.max_access_size = 4,
};

static void g233_spi_instance_init(Object *obj)
{
    G233SpiState *s = G233_SPI(obj);

    memory_region_init_io(&s->iomem, obj, &g233_spi_ops, s,
                          TYPE_G233_SPI, G233_SPI_MMIO_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &s->iomem);
    sysbus_init_irq(SYS_BUS_DEVICE(obj), &s->irq);
}

static const TypeInfo g233_spi_type_info = {
    .parent = TYPE_SYS_BUS_DEVICE,
    .name = TYPE_G233_SPI,
    .instance_size = sizeof(G233SpiState),
    .class_init = g233_spi_class_init,
    .instance_init = g233_spi_instance_init,
};

static void g233_spi_register_type(void)
{
    type_register_static(&g233_spi_type_info);
}

type_init(g233_spi_register_type)