#include "qemu/osdep.h"
#include "qemu/log.h"
#include "hw/core/irq.h"
#include "hw/gpio/g233_gpio.h"
#include "qom/object.h"

#define LEVEL_TRIG 1U
#define EDGE_TRIG  0U

static void g233_gpio_update_irq(G233GPIOState *g233_gpio)
{
    qemu_set_irq(g233_gpio->irq, g233_gpio->int_stat != 0);
}

static void g233_gpio_update_outputs(G233GPIOState *g233_gpio)
{
    int i;

    for (i = 0; i < G233_GPIO_PINS; i++) {
        bool is_out = extract32(g233_gpio->dir, i, 1);
        bool out_value = extract32(g233_gpio->output, i, 1);

        qemu_set_irq(g233_gpio->output_irq[i], is_out ? out_value : 0);
    }
}

static void g233_gpio_update_stat(G233GPIOState *g233_gpio)
{
    for (int i = 0; i < G233_GPIO_PINS; i++) {
        bool is_out = extract32(g233_gpio->dir, i, 1);
        bool int_en = extract32(g233_gpio->int_en, i, 1);
        bool trig_way = extract32(g233_gpio->int_trig, i, 1);
        bool pole = extract32(g233_gpio->int_pole, i, 1);
        bool out_value = extract32(g233_gpio->output, i, 1);
        bool prev_level = extract32(g233_gpio->pre_level, i, 1);

        if (!is_out || !int_en) {
            continue;
        }

        if (trig_way == LEVEL_TRIG) {
            if (out_value == pole) {
                g233_gpio->int_stat |= (0x1) << i;
            } else {
                g233_gpio->int_stat &= ~((0x1) << i);
            }
        } else {
            if ((prev_level == 0 && out_value == 1 && pole == 1) ||
                (prev_level == 1 && out_value == 0 && pole == 0)) {
                g233_gpio->int_stat |= (0x1) << i;
            }
        }
    }

    g233_gpio->pre_level = g233_gpio->output;
    g233_gpio_update_irq(g233_gpio);
}

static uint64_t g233_gpio_read(void *opaque, hwaddr offset, unsigned int size)
{
    G233GPIOState *g233_gpio = G233_GPIO(opaque);
    uint64_t r = 0;
    uint32_t val = 0;
    switch (offset) {
        case G233_GPIO_REG_DIR:
            r = g233_gpio->dir;
            break;
        case G233_GPIO_REG_OUT:
            r = g233_gpio->output;
            break;
        case G233_GPIO_REG_IN:
            for(int i = 0; i < G233_GPIO_PINS; i++) {
                bool is_out = extract32(g233_gpio->dir, i, 1);
                if(is_out) {
                    val = extract32(g233_gpio->output, i, 1);
                } else {
                    val = extract32(g233_gpio->input, i, 1);
                }
                r = deposit32(r, i, 1, val);
            }
            break;
        case G233_GPIO_REG_IE:
            r = g233_gpio->int_en;
            break;
        case G233_GPIO_REG_IS:
            r = g233_gpio->int_stat;
            break;
        case G233_GPIO_REG_TRIG:
            r = g233_gpio->int_trig;
            break;
        case G233_GPIO_REG_POL:
            r = g233_gpio->int_pole;
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR, "%s: bad read offset 0x%" HWADDR_PRIx "\n", __func__, offset);
            break;
    }
    return r;
}

static void g233_gpio_write(void *opaque, hwaddr offset, uint64_t value, unsigned int size)
{
    G233GPIOState *g233_gpio = G233_GPIO(opaque);
    bool update_output_lines = false;

    switch (offset) {
       case G233_GPIO_REG_DIR:
            g233_gpio->dir = value;
            update_output_lines = true;
            break;
        case G233_GPIO_REG_OUT:
            g233_gpio->output = value;
            update_output_lines = true;
            break;
        case G233_GPIO_REG_IE:
            g233_gpio->int_en = value;
            break;
        case G233_GPIO_REG_IS:
            g233_gpio->int_stat &= ~value;
            break;
        case G233_GPIO_REG_TRIG:
            g233_gpio->int_trig = value;
            break;
        case G233_GPIO_REG_POL:
            g233_gpio->int_pole = value;
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR, "%s: bad write offset 0x%" HWADDR_PRIx "\n", __func__, offset);
            break;
    }

    if (update_output_lines) {
        g233_gpio_update_outputs(g233_gpio);
    }

    g233_gpio_update_stat(g233_gpio);
}

static const MemoryRegionOps gpio_ops = {
    .read =  g233_gpio_read,
    .write = g233_gpio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl.min_access_size = 4,
    .impl.max_access_size = 4,
};

static void g233_gpio_set(void *opaque, int line, int value)
{
    G233GPIOState *g233_gpio = G233_GPIO(opaque);

    assert(line >= 0 && line < G233_GPIO_PINS);
    bool int_en = (g233_gpio->int_en >> line) & 0x1;
    bool hi_trig = (g233_gpio->int_pole >> line) & 0x1;
    bool level_trig = (g233_gpio->int_trig >> line) & 0x1;
    bool prev_level = (g233_gpio->input >> line) & 0x1;
    uint32_t int_stat = g233_gpio->int_stat;
    g233_gpio->input = deposit32(g233_gpio->input, line, 1, value);

    if (!int_en) {
        g233_gpio_update_irq(g233_gpio);
        return;
    }

    int_stat = int_stat | (0x1 << line);

    if (((hi_trig && prev_level == 0 && value == 1) ||
         (!hi_trig && prev_level == 1 && value == 0)) && !level_trig) {
        g233_gpio->int_stat = int_stat;
    } else if (level_trig) {
        if ((hi_trig && value == 1) || (!hi_trig && value == 0)) {
            g233_gpio->int_stat = int_stat;
        } else {
            g233_gpio->int_stat &= ~((uint32_t)1 << line);
        }
    }

    g233_gpio_update_irq(g233_gpio);
}

static void g233_gpio_realize(DeviceState *dev, Error **errp)
{
    /* do nothing*/
}

static void g233_gpio_instance_init(Object* obj)
{
    G233GPIOState *g233_gpio = G233_GPIO(obj);
    DeviceState *dev = DEVICE(obj);

    memory_region_init_io(&g233_gpio->mmio, obj, &gpio_ops, g233_gpio,
            TYPE_G233_GPIO, G233_GPIO_SIZE);

    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &g233_gpio->mmio);

    sysbus_init_irq(SYS_BUS_DEVICE(obj), &g233_gpio->irq);

    qdev_init_gpio_in(dev, g233_gpio_set, G233_GPIO_PINS);
    qdev_init_gpio_out(dev, g233_gpio->output_irq, G233_GPIO_PINS);
}
static void g233_gpio_reset(DeviceState *dev)
{
    G233GPIOState *g233_gpio = G233_GPIO(dev);
    g233_gpio->dir      = 0;
    g233_gpio->output   = 0;
    g233_gpio->input    = 0;
    g233_gpio->int_en   = 0;
    g233_gpio->int_stat = 0;
    g233_gpio->int_trig = 0;
    g233_gpio->int_pole = 0;
    g233_gpio->pre_level= 0;

    g233_gpio_update_outputs(g233_gpio);
    g233_gpio_update_irq(g233_gpio);
}

static void g233_gpio_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    dc->realize = g233_gpio_realize;
    device_class_set_legacy_reset(dc, g233_gpio_reset);
}


static const TypeInfo g233_gpio_info = {
    .name = TYPE_G233_GPIO,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(G233GPIOState),
    .class_init = g233_gpio_class_init,
    .instance_init = g233_gpio_instance_init
};

static void g233_gpio_type_register(void) 
{
    type_register_static(&g233_gpio_info);
}

type_init(g233_gpio_type_register);