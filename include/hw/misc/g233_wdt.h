#ifndef G233_WDT_H
#define G233_WDT_H

#include "hw/core/sysbus.h"
#include "qom/object.h"
#include "qemu/timer.h"
#include <stdbool.h>
#include <stdint.h>
#include "hw/core/irq.h"

#define TYPE_G233_WDT "g233-wdt"
typedef struct G233WdtState G233WdtState;
DECLARE_INSTANCE_CHECKER(G233WdtState, G233_WDT, TYPE_G233_WDT)

#define G233_WDT_MMIO_SIZE 0xFFF

#define G233_WDT_CTRL_REG 0x00
#define G233_WDT_LOAD_REG 0x04
#define G233_WDT_VAL_REG  0x08
#define G233_WDT_SR_REG   0x10
#define G233_WDT_KEY_REG  0x0C

#define G233_WDT_CTRL_EN    (1u << 0)
#define G233_WDT_CTRL_INTEN (1u << 1)
#define G233_WDT_CTRL_RSTEN (1u << 2)
#define G233_WDT_CTRL_LOCK  (1u << 3)
#define G233_WDT_CTRL_RW_MASK \
    (G233_WDT_CTRL_EN | G233_WDT_CTRL_INTEN | G233_WDT_CTRL_RSTEN)

#define G233_WDT_SR_TIMEOUT (1u << 0)

#define G233_WDT_FEED_KEY 0x5A5A5A5AU
#define G233_WDT_LOCK_KEY 0x1ACCE551U
#define G233_WDT_LOAD_RESET 0x0000FFFFU

struct G233WdtState {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;
    QEMUTimer timer;

    uint32_t ctrl;
    uint32_t load;
    uint32_t val;
    uint32_t sr;
    uint32_t key;
    bool locked;
};

#endif
