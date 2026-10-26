#!/usr/bin/env python3
"""Build and package best-effort HIP test binaries for the release workflow."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    windows = platform.system() == 'Windows'
    version = os.environ['RELEASE_VERSION']
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_' for c in version):
        raise ValueError('version must be an archive-safe version string')
    rocm_version = os.environ['ROCM_VERSION']
    targets = os.environ['GPU_TARGETS']
    if targets == 'release':
        # Match the tested ROCm release coverage, including Strix Halo.
        common = 'gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201'
        targets = common + ';gfx1153' if windows else 'gfx908;gfx90a;gfx942;gfx950;' + common
    if not targets or any(not arch.startswith('gfx') or not arch.isalnum() for arch in targets.split(';')):
        raise ValueError('GPU_TARGETS must be semicolon-separated gfx architecture names')
    env = os.environ.copy()

    def run(command, name):
        print('+', subprocess.list2cmdline([str(x) for x in command]), flush=True)
        with (logs / (name + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(x) for x in command], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line)
                if name != 'build' or line.startswith('[') or 'error:' in line:
                    print(line, end='', flush=True)
            if process.wait():
                log.flush()
                print(''.join((logs / (name + '.log')).read_text(encoding='utf-8').splitlines(keepends=True)[-80:]), flush=True)
                raise RuntimeError(f'{name} failed: {process.returncode}; see {log.name}')

    run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--index-url',
         'https://stable.repo.amd.com/rocm/whl-next/',
         f'rocm[libraries,devel]=={rocm_version}'], 'install-rocm')
    run(['rocm-sdk', 'init'], 'init-rocm')
    rocm = Path(subprocess.check_output(['rocm-sdk', 'path', '--root'], text=True).strip())
    rocm_bin = subprocess.check_output(['rocm-sdk', 'path', '--bin'], text=True).strip()
    rocm_cmake = subprocess.check_output(['rocm-sdk', 'path', '--cmake'], text=True).strip()
    env.update(ROCM_PATH=str(rocm), HIP_PATH=str(rocm), HIP_PLATFORM='amd',
               CMAKE_PREFIX_PATH=rocm_cmake, PATH=rocm_bin + os.pathsep + env['PATH'])
    if not windows:
        env['LD_LIBRARY_PATH'] = str(rocm / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    else:
        redist = Path(os.environ['VCToolsRedistDir']) / 'x64/Microsoft.VC143.CRT'
        for name in ('vcruntime140.dll', 'msvcp140.dll'):
            if not (redist / name).is_file():
                raise RuntimeError(f'missing compiler redistributable: {redist / name}')
    build = output / 'build'
    command = ['cmake', '-S', source, '-B', build, '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DENGINE_ENABLE_HIP=ON',
               '-DENGINE_ENABLE_CUDA=OFF', '-DENGINE_ENABLE_VULKAN=OFF',
               '-DENGINE_ENABLE_NATIVE_CPU=OFF',
               f'-DENGINE_ENABLE_CPU_ALL_VARIANTS={"ON" if windows else "OFF"}',
               '-DENGINE_ENABLE_OPENMP=OFF', '-DGGML_OPENMP=OFF',
               '-DAUDIOCPP_DEPLOYMENT_BUILD=ON', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON',
               '-DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF', '-DAUDIOCPP_MODEL_SET=full',
               '-DAUDIOCPP_BUILD_C_API=OFF', '-DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF',
               '-DENGINE_ENABLE_CUDA_GRAPHS=ON', f'-DAUDIOCPP_VERSION={version.removeprefix("v")}',
               f'-DGPU_TARGETS={targets}', f'-DCMAKE_PREFIX_PATH={rocm_cmake}']
    if windows:
        llvm = rocm / 'lib' / 'llvm' / 'bin'
        env['HIP_DEVICE_LIB_PATH'] = str(rocm / 'lib' / 'llvm' / 'amdgcn' / 'bitcode')
        env['LLVM_PATH'] = str(rocm / 'lib' / 'llvm')
        command += [f'-DCMAKE_C_COMPILER={llvm / "clang.exe"}',
                    f'-DCMAKE_CXX_COMPILER={llvm / "clang++.exe"}',
                    '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types',
                    # cpp-httplib uses MSVC integer spellings on Windows.
                    f'-DCMAKE_CXX_FLAGS=-isystem "{(source / "external/cpp-httplib").as_posix()}"',
                    '-DGGML_HIP_NO_VMM=ON']
    else:
        llvm = Path(subprocess.check_output([str(Path(rocm_bin) / 'hipconfig'), '-l'], env=env, text=True).strip())
        # Keep the tested linked-backend build; Fish Audio calls a HIP symbol directly.
        command += [f'-DCMAKE_HIP_COMPILER={llvm / "clang"}',
                    '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
                    '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON']
    if shutil.which('ccache'):
        command += ['-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                    '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache']
    if windows:
        run(['pwsh', '-NoProfile', '-File', source / 'scripts/build_windows_hip.ps1',
             '-RocmPath', rocm, '-GpuTargets', targets, '-BuildDir', build,
             '-BuildType', 'Release', '-NoNativeCpu', '-DeploymentBuild',
             '-NativeModelManager', '-Graphs', '-ConfigureOnly'], 'configure-hip')
    run(command, 'configure')
    run(['cmake', '--build', build, '--parallel', str(min(os.cpu_count() or 2, 16)),
         '--target', 'audiocpp_cli', 'audiocpp_server', 'audiocpp_gguf'], 'build')
    shutil.copy2(build / 'CMakeCache.txt', logs / 'CMakeCache.txt')
    binary_dir = build / 'bin'
    if not (list(binary_dir.glob('ggml-hip*.dll')) if windows else list(build.rglob('libggml-hip.a'))):
        raise RuntimeError('HIP backend library was not produced')
    suffix = 'windows-x64' if windows else 'ubuntu-x64'
    stage = output / f'audio-{version}-bin-{suffix}-rocm{rocm_version}-test'
    stage.mkdir()
    for path in binary_dir.iterdir():
        runtime_library = path.suffix == '.dll' if windows else '.so' in path.name
        if path.is_file() and (runtime_library or path.name in (
                'audiocpp_cli', 'audiocpp_server', 'audiocpp_gguf',
                'audiocpp_cli.exe', 'audiocpp_server.exe', 'audiocpp_gguf.exe')):
            shutil.copy2(path, stage / path.name, follow_symlinks=False)
    if windows:
        for name in ('amdhip64_7.dll', 'rocm_kpack.dll', 'amd_comgr.dll'):
            shutil.copy2(Path(rocm_bin) / name, stage / name)
        for pattern in ('vcruntime140*.dll', 'msvcp140*.dll'):
            for dll in redist.glob(pattern):
                shutil.copy2(dll, stage / dll.name)
    else:
        for library in stage.glob('*.so*'):
            run(['patchelf', '--set-rpath', '$ORIGIN', library], 'rpath-' + library.name)
    shutil.copytree(source / 'model_specs', stage / 'model_specs')
    shutil.copytree(source / 'tools', stage / 'tools', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(source / 'LICENSE', stage)
    manifest = dict(source_repository=os.environ.get('SOURCE_REPOSITORY', '0xShug0/audio.cpp'),
                    source_sha=subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip(),
                    rocm_version=rocm_version, gpu_targets=targets.split(';'),
                    cpu='runtime dispatch' if windows else 'AVX2/FMA/F16C',
                    hip_backend='dynamic' if windows else 'linked', deployment=True, full_model_build=True)
    (stage / 'BUILD.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    readme = Path(__file__).with_name('hip_release_README.md').read_text(encoding='utf-8')
    (stage / 'README.md').write_text(readme.replace('@ROCM_VERSION@', rocm_version), encoding='utf-8')
    if windows:
        archive = output / (stage.name + '.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as pack:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    pack.write(path, path.relative_to(stage))
    else:
        archive = output / (stage.name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as pack:
            for path in sorted(stage.iterdir()):
                pack.add(path, arcname=path.name)
    with archive.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    (output / 'SHA256SUMS').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
