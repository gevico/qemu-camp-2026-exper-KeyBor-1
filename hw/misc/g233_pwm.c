#include "qemu/osdep.h"
#include "hw/core/qdev.h"
#include "hw/core/sysbus.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "system/memory.h"
#include "qemu/log.h"
#include "hw/misc/g233_pwm.h"
#include <string.h>

#define TICK_NS 1e6
static void g233_pwm_timer_int(void *opaque);

static void g233_pwm_realize(DeviceState *dev, Error **errp)
{
    G233PwmState *s = G233_PWM(dev);

    for (int i = 0; i < G233_PWM_CHN; i++) {
        s->timer_ctx[i].pwm = s;
        s->timer_ctx[i].channel_index = i;
        timer_init_ns(&s->timer[i], QEMU_CLOCK_VIRTUAL,
                      g233_pwm_timer_int, &s->timer_ctx[i]);
    }
}

static void g233_pwm_timer_int(void *opaque)
{
    G233PwmTimerContext *ctx = opaque;
    G233PwmState *s = ctx->pwm;
    int i = ctx->channel_index;
    uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    uint64_t tick_ns = TICK_NS;

    if (!(s->pwm_ctrl[i] & 0x1)) {
        return;
    }

    if (s->pwm_period[i] == 0) {
        s->pwm_cnt[i] = 0;
        timer_mod(&s->timer[i], now + tick_ns);
        return;
    }

    s->pwm_cnt[i]++;

    /* Period completed: reset counter, set DONE flag, raise IRQ */
    if (s->pwm_cnt[i] >= s->pwm_period[i]) {
        s->pwm_cnt[i] = 0;
        s->pwm_glb |= (1u << (4 + i));          /* CHn_DONE bit */
        qemu_set_irq(s->irq, !!(s->pwm_glb & 0xF0));
    }

    /* Re-arm for the next tick */
    timer_mod(&s->timer[i], now + tick_ns);
}

static void g233_pwm_reset(DeviceState *dev)
{
    G233PwmState *g233_pwm = G233_PWM(dev);
    memset(g233_pwm->pwm_cnt, 0, sizeof(g233_pwm->pwm_cnt));
    memset(g233_pwm->pwm_ctrl, 0, sizeof(g233_pwm->pwm_ctrl));
    memset(g233_pwm->pwm_duty, 0, sizeof(g233_pwm->pwm_duty));
    memset(g233_pwm->pwm_period, 0, sizeof(g233_pwm->pwm_period));
    memset(&g233_pwm->pwm_glb, 0, sizeof(g233_pwm->pwm_glb));
    for (int i = 0; i < G233_PWM_CHN; i++) {
        timer_del(&g233_pwm->timer[i]);
    }
}

static void g233_pwm_unrealize(DeviceState *dev)
{
    G233PwmState *s = G233_PWM(dev);

    for (int i = 0; i < G233_PWM_CHN; i++) {
        timer_del(&s->timer[i]);
    }
}

static void g233_pwm_class_init(ObjectClass *kclass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(kclass);
    dc->realize   = g233_pwm_realize;
    dc->unrealize = g233_pwm_unrealize;
    device_class_set_legacy_reset(dc, g233_pwm_reset);
}

static uint64_t g233_pwm_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    G233PwmState *g233_pwm = G233_PWM(opaque);
    uint32_t value = 0;

    if (size != 4 || (addr & 0x3)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-pwm: invalid read size/alignment addr=0x%" HWADDR_PRIx
                      " size=%u\n",
                      addr, size);
        return 0;
    }

    if (addr >= G233_PWM_CTRL_START && addr < G233_PWM_CTRL_START + 4 * 0x10) {
        uint32_t reg_type = addr % G233_PWM_PER_CHN_SIZE;
        uint32_t chn_num = addr / G233_PWM_PER_CHN_SIZE - 1;

        if (chn_num >= G233_PWM_CHN) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "g233-pwm: channel out of range on read addr=0x%" HWADDR_PRIx
                          " chn=%u\n",
                          addr, chn_num);
            return 0;
        }

        switch (reg_type) {
            case G233_PWM_CTRL_REG:
                value = g233_pwm->pwm_ctrl[chn_num];
                break;
            case G233_PWM_PERIOD_REG:
                value = g233_pwm->pwm_period[chn_num];
                break;
            case G233_PWM_DUTY_REG:
                value = g233_pwm->pwm_duty[chn_num];
                break;
            case G233_PWM_CNT_REG:
                value = g233_pwm->pwm_cnt[chn_num];
                break;
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "g233-pwm: invalid channel register on read addr=0x%" HWADDR_PRIx
                              " reg=0x%x\n",
                              addr, reg_type);
                return 0;
        }
    } else {
        value = g233_pwm->pwm_glb;
    }

    return value;

}

static void g233_pwm_start_timer(G233PwmState *pwm, int chn_num)
{
    uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    uint64_t tick_ns = TICK_NS;

    if (pwm->pwm_ctrl[chn_num] & 0x1) {
        /* Channel just enabled: fire first tick after one interval */
        timer_mod(&pwm->timer[chn_num], now + tick_ns);
    } else {
        /* Channel disabled: cancel any pending tick */
        timer_del(&pwm->timer[chn_num]);
    }
}
static void g233_pwm_mmio_write(void *opaque, hwaddr addr, uint64_t data, unsigned size)
{
    G233PwmState *g233_pwm = G233_PWM(opaque);

    if (size != 4 || (addr & 0x3)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "g233-pwm: invalid write size/alignment addr=0x%" HWADDR_PRIx
                      " size=%u data=0x%" PRIx64 "\n",
                      addr, size, data);
        return;
    }

    if (addr >= G233_PWM_CTRL_START && addr < G233_PWM_CTRL_START + 4 * 0x10) {
        uint32_t reg_type = addr % G233_PWM_PER_CHN_SIZE;
        uint32_t chn_num = addr / G233_PWM_PER_CHN_SIZE - 1;

        if (chn_num >= G233_PWM_CHN) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "g233-pwm: channel out of range addr=0x%" HWADDR_PRIx
                          " chn=%u data=0x%" PRIx64 "\n",
                          addr, chn_num, data);
            return;
        }

        switch (reg_type) {
            case G233_PWM_CTRL_REG:
                if((data & 0x1) ^ (g233_pwm->pwm_glb & (0x1 << chn_num))) {
                    g233_pwm->pwm_glb = (g233_pwm->pwm_glb & 0xF0) | ((data & 0x1) << chn_num);
                }
                g233_pwm->pwm_ctrl[chn_num] = (uint32_t)data;
                g233_pwm_start_timer(g233_pwm, chn_num);
                break;
            case G233_PWM_PERIOD_REG:
                g233_pwm->pwm_period[chn_num] = (uint32_t)data;
                break;
            case G233_PWM_DUTY_REG:
                g233_pwm->pwm_duty[chn_num] = (uint32_t)data;
                break;
            case G233_PWM_CNT_REG:
                g233_pwm->pwm_cnt[chn_num] = (uint32_t)data;
                break;
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "g233-pwm: invalid channel register addr=0x%" HWADDR_PRIx
                              " reg=0x%x data=0x%" PRIx64 "\n",
                              addr, reg_type, data);
                break;
        }
    } else {
        /* DONE bits are W1C in GLB[7:4]. */
        g233_pwm->pwm_glb &= ~((uint32_t)data & 0xF0);
        qemu_set_irq(g233_pwm->irq, !!(g233_pwm->pwm_glb & 0xF0));
    }
}

static const MemoryRegionOps g233_pwm_ops = {
    .write = g233_pwm_mmio_write,
    .read = g233_pwm_mmio_read,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl.min_access_size = 4,
    .impl.max_access_size = 4,
};

static void g233_pwm_instance_init(Object *obj)
{
    G233PwmState *g233_pwm = G233_PWM(obj);

    memory_region_init_io(&g233_pwm->iomem, obj, &g233_pwm_ops, g233_pwm,
            TYPE_G233_PWM, G233_PWM_MMIO_SIZE);

    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &g233_pwm->iomem);

    sysbus_init_irq(SYS_BUS_DEVICE(obj), &g233_pwm->irq);
}

static const TypeInfo g233_pwm_type_info = {
    .parent = TYPE_SYS_BUS_DEVICE,
    .name = TYPE_G233_PWM,
    .instance_size = sizeof(G233PwmState),
    .class_init = g233_pwm_class_init,
    .instance_init = g233_pwm_instance_init
};

static void g233_pwm_register_type(void)
{
    type_register_static(&g233_pwm_type_info);
}

type_init(g233_pwm_register_type)