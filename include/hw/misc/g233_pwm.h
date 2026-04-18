#ifndef G233_PWM_H
#define G233_PWM_H

#include "hw/core/clock.h"
#include "hw/core/sysbus.h"
#include "hw/core/irq.h"
#include "qemu/timer.h"
#include "qemu/typedefs.h"
#include <stdint.h>


#define TYPE_G233_PWM "g233-pwm"

#define G233_PWM(obj) \
    OBJECT_CHECK(G233PwmState, (obj), TYPE_G233_PWM)

#define G233_PWM_CHN             4
#define G233_PWM_MMIO_SIZE       0xFFF
#define G233_PWM_CTRL_START      0x10
#define G233_PWM_REG_SIZE        0x4
#define G233_PWM_CTRL_REG        0x00
#define G233_PWM_PERIOD_REG      0x04
#define G233_PWM_DUTY_REG        0x08
#define G233_PWM_CNT_REG         0x0C
#define G233_PWM_GLB_REG         0x00
#define G233_PWM_PER_CHN_SIZE    0x10

/* Forward declaration so G233PwmTimerContext can be used inside G233PwmState */
typedef struct G233PwmState G233PwmState;

typedef struct {
    G233PwmState *pwm;
    int channel_index;
} G233PwmTimerContext;

struct G233PwmState {
    SysBusDevice parent;

    MemoryRegion iomem;

    qemu_irq irq;

    QEMUTimer timer[G233_PWM_CHN];
    G233PwmTimerContext timer_ctx[G233_PWM_CHN];

    uint32_t pwm_ctrl[G233_PWM_CHN];
    uint32_t pwm_period[G233_PWM_CHN];
    uint32_t pwm_duty[G233_PWM_CHN];
    uint32_t pwm_cnt[G233_PWM_CHN];

    uint32_t pwm_glb;
};

#endif