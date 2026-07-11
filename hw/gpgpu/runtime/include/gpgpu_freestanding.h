/*
 * Small compatibility shim for freestanding bare-metal builds.
 *
 * Hosted builds use the system C library headers. Some packaged
 * riscv64-unknown-elf toolchains provide compiler/binutils without
 * newlib/picolibc headers, so keep this runtime usable without them.
 */
#ifndef GPGPU_FREESTANDING_H
#define GPGPU_FREESTANDING_H

#include <stddef.h>

#if defined(__has_include)
#if __has_include(<errno.h>)
#include <errno.h>
#endif
#if __has_include(<string.h>)
#include <string.h>
#else
void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
#endif
#else
#include <errno.h>
#include <string.h>
#endif

#ifndef EIO
#define EIO 5
#endif
#ifndef ENOMEM
#define ENOMEM 12
#endif
#ifndef ENODEV
#define ENODEV 19
#endif
#ifndef EINVAL
#define EINVAL 22
#endif
#ifndef ERANGE
#define ERANGE 34
#endif
#ifndef ENOTSUP
#define ENOTSUP 95
#endif

#endif /* GPGPU_FREESTANDING_H */
