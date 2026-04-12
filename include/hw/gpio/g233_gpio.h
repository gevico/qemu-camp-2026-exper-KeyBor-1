#ifndef G233_GPIO_H
#define G233_GPIO_H

#include "hw/core/sysbus.h"
#include "qom/object.h"

#define TYPE_G233_GPIO "g233_machine.gpio"
typedef struct G233GPIOState G233GPIOState;
DECLARE_INSTANCE_CHECKER(G233GPIOState, G233_GPIO,
                         TYPE_G233_GPIO)

#define G233_GPIO_PINS 32

#define G233_GPIO_SIZE 0xFF

#define G233_GPIO_REG_DIR     0x000
#define G233_GPIO_REG_OUT     0x004
#define G233_GPIO_REG_IN      0x008
#define G233_GPIO_REG_IE      0x00C
#define G233_GPIO_REG_IS      0x010
#define G233_GPIO_REG_TRIG    0x014
#define G233_GPIO_REG_POL     0x018


struct G233GPIOState {
    SysBusDevice parent_obj;
    MemoryRegion mmio;

    qemu_irq irq;
    qemu_irq output_irq[G233_GPIO_PINS];

    uint32_t dir;
    uint32_t output;
    uint32_t input;
    uint32_t int_en;
    uint32_t int_stat;
    uint32_t int_trig;
    uint32_t int_pole;
    uint32_t pre_level;
};
#endif
