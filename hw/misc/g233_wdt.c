#include "qemu/osdep.h"
#include "hw/core/qdev.h"
#include "qemu/timer.h"
#include "system/memory.h"
#include "qemu/log.h"
#include "hw/misc/g233_wdt.h"

#define TICK_NS 1e6
static void g233_wdt_timer_ctrl(G233WdtState *s)
{
    if(s->ctrl & G233_WDT_CTRL_EN) {
        s->val = s->load;
        uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        timer_mod(&s->timer, now + TICK_NS);
    } else {
        timer_del(&s->timer);
    }
}
static void g233_wdt_timer_cb(void *opaque)
{
    G233WdtState *s = opaque;
    s->val--;
    if(s->val <= 0U) {
        s->val = 0;
        if(s->ctrl & G233_WDT_CTRL_INTEN) {
            qemu_set_irq(s->irq, 1);
            s->ctrl &= ~G233_WDT_CTRL_RW_MASK;
        } else if(s->ctrl & G233_WDT_CTRL_RSTEN) {
            qemu_set_irq(s->irq, 1);
            s->ctrl &= ~G233_WDT_CTRL_RW_MASK;
        }
        s->sr |= G233_WDT_SR_TIMEOUT;
        timer_del(&s->timer);
    } else {
        uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        timer_mod(&s->timer, now + TICK_NS);
    }
}

static void g233_wdt_feed(G233WdtState *s)
{
    if(!(s->ctrl & G233_WDT_CTRL_EN)) return;
    s->val = s->load;
    s->sr &= ~G233_WDT_SR_TIMEOUT;
    uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    timer_mod(&s->timer, now + TICK_NS);
}

static void g233_wdt_reset(DeviceState *dev)
{
    G233WdtState *s = G233_WDT(dev);

    s->ctrl = 0;
    s->load = G233_WDT_LOAD_RESET;
    s->val = G233_WDT_LOAD_RESET;
    s->sr = 0;
    s->key = 0;
    s->locked = false;

    timer_del(&s->timer);
    qemu_set_irq(s->irq, 0);
}

static void g233_wdt_realize(DeviceState *dev, Error **errp)
{
    G233WdtState *s = G233_WDT(dev);

    timer_init_ns(&s->timer, QEMU_CLOCK_VIRTUAL, g233_wdt_timer_cb, s);
}

static void g233_wdt_unrealize(DeviceState *dev)
{
    G233WdtState *s = G233_WDT(dev);

    timer_del(&s->timer);
}

static void g233_wdt_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = g233_wdt_realize;
    dc->unrealize = g233_wdt_unrealize;
    device_class_set_legacy_reset(dc, g233_wdt_reset);
}

static uint64_t g233_wdt_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    G233WdtState *s = G233_WDT(opaque);

    if (size != 4 || (addr & 0x3)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-wdt: invalid read size/alignment addr=0x%" HWADDR_PRIx
                      " size=%u\n",
                      addr, size);
        return 0;
    }

    switch (addr) {
    case G233_WDT_CTRL_REG:
        return s->ctrl;
    case G233_WDT_LOAD_REG:
        return s->load;
    case G233_WDT_VAL_REG:
        return s->val;
    case G233_WDT_SR_REG:
        return s->sr & G233_WDT_SR_TIMEOUT;
    case G233_WDT_KEY_REG:
        return 0;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-wdt: invalid read addr=0x%" HWADDR_PRIx "\n",
                      addr);
        return 0;
    }
}

static void g233_wdt_mmio_write(void *opaque, hwaddr addr,
                                uint64_t data, unsigned size)
{
    G233WdtState *s = G233_WDT(opaque);
    uint32_t value = (uint32_t)data;

    if (size != 4 || (addr & 0x3)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-wdt: invalid write size/alignment addr=0x%" HWADDR_PRIx
                      " size=%u data=0x%" PRIx64 "\n",
                      addr, size, data);
        return;
    }

    switch (addr) {
    case G233_WDT_CTRL_REG:
        if (!s->locked) {
            s->ctrl = value & G233_WDT_CTRL_RW_MASK;
            g233_wdt_timer_ctrl(s);
        }
        break;
    case G233_WDT_LOAD_REG:
        if (!s->locked) {
            s->load = value;
        }
        break;
    case G233_WDT_VAL_REG:
        qemu_log_mask(LOG_GUEST_ERROR, "g233-wdt: WDT_VAL is read-only\n");
        break;
    case G233_WDT_SR_REG:
        s->sr &= ~(value & G233_WDT_SR_TIMEOUT);
        break;
    case G233_WDT_KEY_REG:
        s->key = value;
        if (value == G233_WDT_FEED_KEY) {
            g233_wdt_feed(s);
        } else if (value == G233_WDT_LOCK_KEY) {
            s->locked = true;
        }
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-wdt: invalid write addr=0x%" HWADDR_PRIx
                      " data=0x%" PRIx64 "\n",
                      addr, data);
        break;
    }
}

static const MemoryRegionOps g233_wdt_ops = {
    .read = g233_wdt_mmio_read,
    .write = g233_wdt_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl.min_access_size = 4,
    .impl.max_access_size = 4,
};

static void g233_wdt_instance_init(Object *obj)
{
    G233WdtState *s = G233_WDT(obj);

    memory_region_init_io(&s->iomem, obj, &g233_wdt_ops, s,
                          TYPE_G233_WDT, G233_WDT_MMIO_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &s->iomem);
    sysbus_init_irq(SYS_BUS_DEVICE(obj), &s->irq);
}


static void g233_wdt_type_register(void)
{
    static const TypeInfo g233_wdt_type_info = {
        .parent = TYPE_SYS_BUS_DEVICE,
        .name = TYPE_G233_WDT,
        .instance_size = sizeof(G233WdtState),
        .class_init = g233_wdt_class_init,
        .instance_init = g233_wdt_instance_init,
    };
    type_register_static(&g233_wdt_type_info);
}

type_init(g233_wdt_type_register)

