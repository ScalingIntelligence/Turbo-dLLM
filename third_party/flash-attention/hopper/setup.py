# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import ast
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.version import Version, parse
from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


# ninja build does not work unless include_dirs are absolute paths.
this_dir = os.path.dirname(os.path.abspath(__file__))

with open(Path(this_dir).parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


PACKAGE_NAME = "flash_attn_3"
DISTRIBUTION_NAME = "bdlm-flash-attn-3"
BDLM_VARIANT = "bdlm-cp-splitd-v3"
BUILD_METADATA_FORMAT = "bdlm.flash_attn_3.build.v3"
BUILD_METADATA_NAME = "build_metadata.json"
REQUIRED_BDLM_OPS = (
    "bdlm_fwd_accum",
    "bdlm_ragged_prefix_fwd",
    "bdlm_ragged_prefix_bwd",
    "dflash_interval_fwd",
    "dflash_interval_bwd",
)

# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
# ROCm specific settings
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if USE_TRITON_ROCM:
    SKIP_CUDA_BUILD = True

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE"
DISABLE_VARLEN = os.getenv("FLASH_ATTENTION_DISABLE_VARLEN", "FALSE") == "TRUE"
DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"

ENABLE_VCOLMAJOR = os.getenv("FLASH_ATTENTION_ENABLE_VCOLMAJOR", "FALSE") == "TRUE"

DISABLE_HDIMDIFF64 = (
    os.getenv("FLASH_ATTENTION_DISABLE_HDIMDIFF64", "FALSE") == "TRUE"
    or DISABLE_HDIM64
    or DISABLE_HDIM256
)
DISABLE_HDIMDIFF192 = (
    os.getenv("FLASH_ATTENTION_DISABLE_HDIMDIFF192", "FALSE") == "TRUE"
    or DISABLE_HDIM192
    or DISABLE_HDIM128
)

# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and emit both sm_80 and sm_89 images from the unchanged '_sm80.cu' templates.
from torch.utils.cpp_extension import (
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True
    else:
        raise ValueError(f"unsupported BUILD_TARGET: {BUILD_TARGET!r}")

if IS_ROCM and not SKIP_CUDA_BUILD:
    raise RuntimeError(
        "bdlm-flash-attn-3 production wheels require CUDA; ROCm builds are not supported"
    )


def build_config():
    return {
        "build_flags": {
            "FLASHATTENTION_DISABLE_BACKWARD": DISABLE_BACKWARD,
            "FLASHATTENTION_DISABLE_SPLIT": DISABLE_SPLIT,
            "FLASHATTENTION_DISABLE_PAGEDKV": DISABLE_PAGEDKV,
            "FLASHATTENTION_DISABLE_APPENDKV": DISABLE_APPENDKV,
            "FLASHATTENTION_DISABLE_LOCAL": DISABLE_LOCAL,
            "FLASHATTENTION_DISABLE_SOFTCAP": DISABLE_SOFTCAP,
            "FLASHATTENTION_DISABLE_PACKGQA": DISABLE_PACKGQA,
            "FLASHATTENTION_DISABLE_FP16": DISABLE_FP16,
            "FLASHATTENTION_DISABLE_FP8": DISABLE_FP8,
            "FLASHATTENTION_DISABLE_VARLEN": DISABLE_VARLEN,
            "FLASHATTENTION_DISABLE_CLUSTER": DISABLE_CLUSTER,
            "FLASHATTENTION_DISABLE_HDIM64": DISABLE_HDIM64,
            "FLASHATTENTION_DISABLE_HDIM96": DISABLE_HDIM96,
            "FLASHATTENTION_DISABLE_HDIM128": DISABLE_HDIM128,
            "FLASHATTENTION_DISABLE_HDIM192": DISABLE_HDIM192,
            "FLASHATTENTION_DISABLE_HDIM256": DISABLE_HDIM256,
            "FLASHATTENTION_DISABLE_SM8x": DISABLE_SM8x,
            "FLASHATTENTION_ENABLE_VCOLMAJOR": ENABLE_VCOLMAJOR,
            "FLASH_ATTENTION_DISABLE_HDIMDIFF64": DISABLE_HDIMDIFF64,
            "FLASH_ATTENTION_DISABLE_HDIMDIFF192": DISABLE_HDIMDIFF192,
        }
    }


def create_build_config_file():
    config = build_config()
    with open(Path(this_dir) / "flash_attn_config.py", "w", encoding="utf-8") as f:
        f.write("# Auto-generated by flash attention 3 setup.py\n")
        f.write(f"CONFIG = {repr(config)}\n")
        f.write("\n")

        f.write("def show():\n")
        f.write("    from pprint import pprint\n")
        f.write("    pprint(CONFIG)\n")
        f.write("\n")


def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
                      ) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            nvcc_from_env = nvcc
        config.append(f'nvcc_from_env = {nvcc_from_env}')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_post_cflags_sm80 = (
            [
                s
                if s != 'arch=compute_90a,code=sm_90a'
                else 'arch=compute_80,code=sm_80'
                for s in cuda_post_cflags
            ]
            + ['-gencode', 'arch=compute_89,code=sm_89']
        )
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        cuda_post_cflags_sm80_sm90 = (
            cuda_post_cflags
            if DISABLE_SM8x
            else cuda_post_cflags
            + [
                '-gencode',
                'arch=compute_80,code=sm_80',
                '-gencode',
                'arch=compute_89,code=sm_89',
            ]
        )
        flags.append(f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule_sm80 = ['rule cuda_compile_sm80'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80'
        ]
        cuda_compile_rule_sm80_sm90 = ['rule cuda_compile_sm80_sm90'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith('_sm90.cu'):
                rule = 'cuda_compile'
            elif source_file.endswith('_sm80.cu'):
                rule = 'cuda_compile_sm80'
            else:
                rule = 'cuda_compile_sm80_sm90'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


# Monkey patching
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} must be built from this checkout, but nvcc was not found. "
        "Use a CUDA devel image with CUDA_HOME set; prebuilt wheel fallback is disabled."
    )


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]


ext_modules = []
BUILD_CUDA_TOOLKIT = None
BUILD_FLASH_API_SOURCE = None
# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
if not USE_TRITON_ROCM:
    cutlass_include = Path(this_dir).parent / "csrc" / "cutlass" / "include"
    if not cutlass_include.is_dir():
        raise RuntimeError(
            "CUTLASS sources are missing. Initialize repository submodules before "
            "building bdlm-flash-attn-3; setup.py does not fetch build inputs."
        )

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    create_build_config_file()
    check_if_cuda_home_none(PACKAGE_NAME)
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    BUILD_CUDA_TOOLKIT = str(bare_metal_version)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("FlashAttention-3 is only supported on CUDA 12.3 and above")
    elif bare_metal_version >= Version("13.0"):
        # CUDA 13.0+ uses system nvcc and CCCL headers are in /usr/local/cuda/include/cccl/
        cccl_include = os.path.join(CUDA_HOME, "include", "cccl")
        for env_var in ["CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"]:
            current = os.environ.get(env_var, "")
            os.environ[env_var] = cccl_include + (":" + current if current else "")

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir).parent
    cutlass_dir = repo_dir / "csrc" / "cutlass"

    feature_args = (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_PAGEDKV"] if DISABLE_PAGEDKV else [])
        + (["-DFLASHATTENTION_DISABLE_SPLIT"] if DISABLE_SPLIT else [])
        + (["-DFLASHATTENTION_DISABLE_APPENDKV"] if DISABLE_APPENDKV else [])
        + (["-DFLASHATTENTION_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DFLASHATTENTION_DISABLE_SOFTCAP"] if DISABLE_SOFTCAP else [])
        + (["-DFLASHATTENTION_DISABLE_PACKGQA"] if DISABLE_PACKGQA else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DFLASHATTENTION_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DFLASHATTENTION_DISABLE_VARLEN"] if DISABLE_VARLEN else [])
        + (["-DFLASHATTENTION_DISABLE_CLUSTER"] if DISABLE_CLUSTER else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if DISABLE_SM8x else [])
        + (["-DFLASHATTENTION_ENABLE_VCOLMAJOR"] if ENABLE_VCOLMAJOR else [])
        + (["-DFLASHATTENTION_DISABLE_HDIMDIFF64"] if DISABLE_HDIMDIFF64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIMDIFF192"] if DISABLE_HDIMDIFF192 else [])
    )

    DTYPE_FWD_SM80 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    DTYPE_FWD_SM90 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else []) + (["e4m3"] if not DISABLE_FP8 else [])
    HALF_DTYPE_FWD_SM90 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    DTYPE_BWD = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS_BWD = (
        []
        + ([64] if not DISABLE_HDIM64 else [])
        + ([96] if not DISABLE_HDIM96 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([192] if not DISABLE_HDIM192 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    # build will now explode with this compilation grouping given all our templating
    # HEAD_DIMENSIONS_FWD = ["all", "diff"]
    HEAD_DIMENSIONS_FWD = HEAD_DIMENSIONS_BWD
    HEAD_DIMENSIONS_DIFF64_FWD = (
        []
        + (["64_256"] if not DISABLE_HDIMDIFF64 else [])
        + (["64_512"] if not DISABLE_HDIMDIFF64 else [])
    )
    HEAD_DIMENSIONS_DIFF192_FWD = (
        []
        + (["192_128"] if not DISABLE_HDIMDIFF192 else [])
    )
    HEAD_DIMENSIONS_FWD_SM80 = HEAD_DIMENSIONS_BWD
    SPLIT = [""] + (["_split"] if not DISABLE_SPLIT else [])
    PAGEDKV = [""] + (["_paged"] if not DISABLE_PAGEDKV else [])
    SOFTCAP = [""] + (["_softcap"] if not DISABLE_SOFTCAP else [])
    SOFTCAP_ALL = [""] if DISABLE_SOFTCAP else ["_softcapall"]
    PACKGQA = [""] + (["_packgqa"] if not DISABLE_PACKGQA else [])
    # We already always hard-code PackGQA=true for Sm8x
    sources_fwd_sm80 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}_sm80.cu"
                        for hdim, dtype, split, paged, softcap in itertools.product(HEAD_DIMENSIONS_FWD_SM80, DTYPE_FWD_SM80, SPLIT, PAGEDKV, SOFTCAP_ALL)]
    # We already always hard-code PackGQA=true for Sm9x if PagedKV or Split
    sources_fwd_sm90 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
                        for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_FWD, DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
                        if not (packgqa and (paged or split))]
    if not DISABLE_HDIMDIFF64:
        sources_fwd_sm90 += [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
                             for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_DIFF64_FWD, HALF_DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
                             if not (packgqa and (paged or split))]
    if not DISABLE_HDIMDIFF192:
        sources_fwd_sm90 += [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
                            for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_DIFF192_FWD, DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
                            if not (packgqa and (paged or split))]
    sources_bwd_sm80 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm80.cu"
                        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP)]
    sources_bwd_sm90 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm90.cu"
                        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP_ALL)]
    if DISABLE_BACKWARD:
        sources_bwd_sm90 = []
        sources_bwd_sm80 = []
    
    # Choose between flash_api.cpp and flash_api_stable.cpp based on torch version
    torch_version = parse(torch.__version__)
    target_version = parse("2.9.0.dev20250830")
    stable_args = []
      
    force_legacy_api = os.environ.get("FLASH_ATTENTION_FORCE_LEGACY_API", "0").lower() in {
        "1",
        "true",
        "on",
        "enabled",
    }
    use_stable_api = os.environ.get("FLASH_ATTENTION_USE_STABLE_API", "0").lower() in {
        "1",
        "true",
        "on",
        "enabled",
    }
    if use_stable_api and torch_version >= target_version and not force_legacy_api:
        flash_api_source = "flash_api_stable.cpp"
        stable_args = ["-DTORCH_TARGET_VERSION=0x0209000000000000"]  # Targets minimum runtime version torch 2.9.0
    else:
        # The BDLM production backend requires all custom operators registered
        # in flash_api.cpp, including ragged prefix forward/backward. Keep the
        # complete API as the default until flash_api_stable.cpp reaches parity.
        flash_api_source = "flash_api.cpp"
    BUILD_FLASH_API_SOURCE = flash_api_source

    sources = (
        [flash_api_source]
        + (sources_fwd_sm80 if not DISABLE_SM8x else []) + sources_fwd_sm90
        + (sources_bwd_sm80 if not DISABLE_SM8x else []) + sources_bwd_sm90
    )
    if not DISABLE_SPLIT:
        sources += ["flash_fwd_combine.cu"]
    sources += ["bdlm_cp_fusion.cu"]
    sources += ["flash_prepare_scheduler.cu"]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--use_fast_math",
        # "--keep",
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        "--resource-usage",  # printing out number of registers
        # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        Path(this_dir),
        cutlass_dir / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DPy_LIMITED_API=0x03090000"] + stable_args + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
            py_limited_api=True,
        )
    )


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    return str(ast.literal_eval(version_match.group(1)))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inputs():
    source_suffixes = {
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".inl",
        ".py",
        ".toml",
    }
    excluded_dirs = {"__pycache__", "build", "dist", ".eggs"}
    paths = []
    for path in Path(this_dir).rglob("*"):
        relative = path.relative_to(this_dir)
        if not path.is_file() or any(part in excluded_dirs for part in relative.parts):
            continue
        if path.name == BUILD_METADATA_NAME or path.name == "flash_attn_config.py":
            continue
        if path.name.startswith("test_") or path.name.startswith("benchmark_"):
            continue
        if path.suffix in source_suffixes:
            paths.append(path)
    cutlass_include = Path(this_dir).parent / "csrc" / "cutlass" / "include"
    paths.extend(
        path
        for path in cutlass_include.rglob("*")
        if path.is_file() and path.suffix in source_suffixes
    )
    return sorted(set(paths), key=lambda path: path.relative_to(Path(this_dir).parent).as_posix())


def _source_metadata():
    digest = hashlib.sha256()
    paths = _source_inputs()
    source_root = Path(this_dir).parent
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    revision = os.getenv("BDLM_FA3_SOURCE_REVISION")
    if not revision:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "revision": revision,
    }


def _python_package_metadata(relative_root):
    relative_root = Path(relative_root)
    package_root = Path(this_dir) / relative_root
    files = {}
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(Path(this_dir)).as_posix()
        files[relative] = _sha256(path)
    if not files:
        raise RuntimeError(f"packaged Python source tree is empty: {package_root}")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "root": relative_root.as_posix(),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def _write_build_metadata(binary_path):
    binary_path = Path(binary_path).resolve()
    interface_path = Path(this_dir) / "flash_attn_interface.py"
    cuda_architectures = (
        ["sm_90a"]
        if DISABLE_SM8x
        else ["sm_80", "sm_89", "sm_90a"]
    )
    metadata = {
        "format": BUILD_METADATA_FORMAT,
        "variant": BDLM_VARIANT,
        "distribution": {
            "name": DISTRIBUTION_NAME,
            "version": get_package_version(),
        },
        "module": f"{PACKAGE_NAME}._C",
        "required_ops": list(REQUIRED_BDLM_OPS),
        "binary": {
            "path": f"{PACKAGE_NAME}/{binary_path.name}",
            "sha256": _sha256(binary_path),
            "extension_suffix": binary_path.name.removeprefix("_C"),
        },
        "python_interface": {
            "path": interface_path.name,
            "sha256": _sha256(interface_path),
        },
        "python_package": _python_package_metadata(
            Path(PACKAGE_NAME) / "bdlm_splitd"
        ),
        "source": _source_metadata(),
        "build": {
            "python": platform.python_version(),
            "python_extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cuda_toolkit": BUILD_CUDA_TOOLKIT,
            "cuda_architectures": cuda_architectures,
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "flash_api_source": BUILD_FLASH_API_SOURCE,
            "feature_flags": build_config()["build_flags"],
        },
    }
    canonical = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    metadata["build_id"] = hashlib.sha256(canonical).hexdigest()
    metadata_path = binary_path.parent / BUILD_METADATA_NAME
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote BDLM FlashAttention build metadata: {metadata_path}")


def _build_splitd_aot():
    output_root = Path(this_dir) / PACKAGE_NAME / "bdlm_splitd/_artifacts"
    build_environment = dict(os.environ)
    build_environment["CUTE_DSL_ARCH"] = "sm_90a"
    subprocess.run(
        [
            sys.executable,
            "-m",
            f"{PACKAGE_NAME}.bdlm_splitd._aot",
            "--output-root",
            str(output_root),
        ],
        cwd=this_dir,
        env=build_environment,
        check=True,
    )


class BDLMBuildExtension(BuildExtension):
    def run(self):
        super().run()
        extension = next(
            (item for item in self.extensions if item.name == f"{PACKAGE_NAME}._C"),
            None,
        )
        if extension is None:
            raise RuntimeError("BDLM FlashAttention native extension was not configured")
        binary_path = Path(self.get_ext_fullpath(extension.name))
        if not binary_path.exists():
            raise RuntimeError(f"BDLM FlashAttention binary was not built: {binary_path}")
        _write_build_metadata(binary_path)


class BDLMWheelCommand(_bdist_wheel):
    def run(self):
        if SKIP_CUDA_BUILD:
            raise RuntimeError(
                "FLASH_ATTENTION_SKIP_CUDA_BUILD may only be used for an sdist; "
                "a BDLM wheel must contain a locally built flash_attn_3._C binary"
            )
        _build_splitd_aot()
        super().run()

# Build install_requires based on platform
if IS_ROCM:
    # Note: torch is excluded because pip resolves it to CUDA PyTorch from PyPI, overwriting any pre-installed ROCm PyTorch. Users must have torch installed.
    install_requires = [
        "einops",
        "packaging",
        "ninja",
    ]
else:
    install_requires = [
        "torch",
        "einops",
        "packaging",
        "cuda-bindings>=12.8,<13",
        "apache-tvm-ffi>=0.1.12,<0.2",
    ]

setup(
    name=DISTRIBUTION_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    py_modules=["flash_attn_interface", "flash_attn_config"],
    package_data={
        PACKAGE_NAME: [
            BUILD_METADATA_NAME,
            "bdlm_splitd/_artifacts/*/manifest.json",
            "bdlm_splitd/_artifacts/*/*/*.so",
            "bdlm_splitd/README.md",
        ]
    },
    include_package_data=True,
    description="BDLM FlashAttention-3 native extension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": BDLMWheelCommand, "build_ext": BDLMBuildExtension}
    if ext_modules
    else {
        "bdist_wheel": BDLMWheelCommand,
    },
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "developer-jit": [
            "ninja",
            "nvidia-cutlass-dsl>=4.5.1,<4.6",
            "torch-c-dlpack-ext>=0.1.5,<0.2",
            "quack-kernels>=0.4.1,<0.5",
        ]
    },
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
    zip_safe=False,
)
