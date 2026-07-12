/*
 * QEMU Educational GPGPU - Backend Registry
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 后端注册表: name → factory 函数的映射。
 * class_init 时初始化，realize 时查表创建实例。
 */

#include "qemu/osdep.h"
#include "../include/gpgpu_backend_comm.h"

typedef struct gpgpu_backend *(*gpgpu_backend_factory)(void);

static GHashTable *backend_registry;

void gpgpu_backend_registry_init(void)
{
    backend_registry = g_hash_table_new(g_str_hash, g_str_equal);

    /* 向后端 factory 注册 */
    g_hash_table_insert(backend_registry, (void *)"rv32",
                        (void *)gpgpu_backend_rv32_new);

    g_hash_table_insert(backend_registry, (void *)"vortex",
                        (void *)gpgpu_backend_vortex_new);
}

struct gpgpu_backend *gpgpu_backend_create(const char *name)
{
    if (!backend_registry || !name) {
        return NULL;
    }

    gpgpu_backend_factory factory =
        g_hash_table_lookup(backend_registry, name);
    if (!factory) {
        return NULL;
    }

    return factory();
}
