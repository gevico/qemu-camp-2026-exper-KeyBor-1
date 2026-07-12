# GPGPU 设备的 Meson 构建配置

gpgpu_inc = include_directories('.')

gpgpu_common_sources = files(
    'gpgpu.c',
    'backend/src/gpgpu_backend_registry.c',
    'backend/src/gpgpu_backend_rv32.c',
    'backend/src/gpgpu_backend_vortex.c',
    'backend/arch/riscv/rvi_interpreter/gpgpu_core.c',
)

# Vortex backend 需要 C++ 编译器 (链接 libsimx.so)
# 如果 C++ 不可用，仅构建 rv32 后端
cpp_available = add_languages('cpp', required: false)

if cpp_available
    # Vortex 是外部项目，不在 QEMU source tree 内。
    # 使用 compile_args 传递头文件路径，避免 meson source-tree 检查。
    vortex_home = meson.project_source_root() / '..' / 'vortex'

    vortex_cpp = files(
        'backend/arch/riscv/vortex_backend/vortex_simx_wrapper.cpp',
    )

    vortex_simx_dir    = vortex_home / 'sim' / 'simx'
    vortex_common_dir  = vortex_home / 'sim' / 'common'
    vortex_hw_build_dir = vortex_home / 'build' / 'hw'
    vortex_lib_dir     = vortex_home / 'build' / 'sim' / 'simx'
    vortex_ramu_dir    = vortex_home / 'third_party' / 'ramulator'

    cpp = meson.get_compiler('cpp')
    vortex_dep = declare_dependency(
        compile_args: [
            '-I' + vortex_simx_dir,
            '-I' + vortex_common_dir,
            '-I' + vortex_hw_build_dir,
            '-DXLEN_32',
        ],
        link_args: [
            '-L' + vortex_lib_dir,
            '-lsimx',
            '-Wl,-rpath,' + vortex_lib_dir,
            '-L' + vortex_ramu_dir,
            '-lramulator',
            '-Wl,-rpath,' + vortex_ramu_dir,
        ],
        dependencies: [cpp.find_library('stdc++', required: true)],
    )

    # 分步添加，保证 include path 正常传播
    system_ss.add(when: 'CONFIG_GPGPU', if_true: gpgpu_common_sources)
    system_ss.add(when: 'CONFIG_GPGPU', if_true: vortex_cpp)
    system_ss.add(when: 'CONFIG_GPGPU', if_true: vortex_dep)
    system_ss.add(when: 'CONFIG_GPGPU', if_true: declare_dependency(include_directories: gpgpu_inc))
    message('GPGPU: vortex backend enabled (libsimx.so)')
else
    system_ss.add(when: 'CONFIG_GPGPU', if_true: gpgpu_common_sources)
    system_ss.add(when: 'CONFIG_GPGPU', if_true: declare_dependency(include_directories: gpgpu_inc))
    message('GPGPU: C++ unavailable, vortex backend disabled (rv32 only)')
endif
