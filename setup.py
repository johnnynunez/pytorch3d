#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# All static metadata (name, version, dependencies, scripts, packages) lives in
# pyproject.toml. This file only contains what cannot be expressed there: the
# C++/CUDA extension build, which has to import torch at install time.

import glob
import os
import sys
import warnings
from typing import List, Optional

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension

# The implicitron trainer lives outside the importable `pytorch3d/` directory
# (at `projects/implicitron_trainer/`) and gets grafted in under the
# `pytorch3d.implicitron_trainer` name. setuptools' declarative `find` cannot
# express that, so we keep this small bit of dynamic config here.
TRAINER = "pytorch3d.implicitron_trainer"
PACKAGES = find_packages(include=["pytorch3d*"], exclude=["tests*"]) + [TRAINER]
PACKAGE_DIR = {TRAINER: "projects/implicitron_trainer"}


def get_existing_ccbin(nvcc_args: List[str]) -> Optional[str]:
    """
    Given a list of nvcc arguments, return the compiler if specified.

    Note from CUDA doc: Single value options and list options must have
    arguments, which must follow the name of the option itself by either
    one of more spaces or an equals character.
    """
    last_arg = None
    for arg in reversed(nvcc_args):
        if arg == "-ccbin":
            return last_arg
        if arg.startswith("-ccbin="):
            return arg[7:]
        last_arg = arg
    return None


def get_extensions():
    no_extension = os.getenv("PYTORCH3D_NO_EXTENSION", "0") == "1"
    if no_extension:
        msg = "SKIPPING EXTENSION BUILD. PYTORCH3D WILL NOT WORK!"
        print(msg, file=sys.stderr)
        warnings.warn(msg)
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pytorch3d", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    extension = CppExtension

    # C++17 flag differs by compiler: MSVC uses /std:c++17, gcc/clang use
    # -std=c++17. torch.utils.cpp_extension does NOT auto-translate this.
    if sys.platform == "win32":
        # CUDA 13.2's CCCL requires MSVC's standard-conforming preprocessor.
        cxx_args = ["/std:c++17", "/Zc:preprocessor"]
    else:
        cxx_args = ["-std=c++17"]
    extra_compile_args = {"cxx": cxx_args}
    define_macros = []
    include_dirs = [extensions_dir]

    # Add a runtime search path so the built `_C.so` / `_C.dylib` can find
    # torch's shared libraries (libtorch_cpu, libc10, libtorch_cuda...) at
    # import time. Without this, after installation the user sees
    # `ImportError: libc10.so: cannot open shared object file` (Linux) or
    # `Library not loaded: @rpath/libc10.dylib` (macOS).
    #
    # Mirrors how torchvision / torchaudio wheels are linked. `$ORIGIN` /
    # `@loader_path` resolve to the directory holding `_C.so`
    # (`site-packages/pytorch3d/`), so `../torch/lib` is the sibling
    # `site-packages/torch/lib/` regardless of the install prefix.
    if sys.platform == "darwin":
        extra_link_args = ["-Wl,-rpath,@loader_path/../torch/lib"]
    elif sys.platform.startswith("linux"):
        extra_link_args = ["-Wl,-rpath,$ORIGIN/../torch/lib"]
    else:
        extra_link_args = []

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    force_no_cuda = os.getenv("PYTORCH3D_FORCE_NO_CUDA", "0") == "1"
    if (
        not force_no_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    ) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        # Thrust is only used for its tuple objects.
        # With CUDA 11.0 we can't use the cudatoolkit's version of cub.
        # We take the risk that CUB and Thrust are incompatible, because
        # we aren't using parts of Thrust which actually use CUB.
        define_macros += [("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        cub_home = os.environ.get("CUB_HOME", None)
        nvcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.name != "nt":
            nvcc_args.append("-std=c++17")
        else:
            # Forward the same preprocessor mode to cl.exe when invoked by nvcc.
            nvcc_args.extend(["-Xcompiler", "/Zc:preprocessor"])

        # CUDA 13.0+ compatibility flags for pulsar.
        # Starting with CUDA 13, __global__ function visibility changed.
        # See: https://developer.nvidia.com/blog/
        #      cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            major = int(cuda_version.split(".")[0])
            if major >= 13:
                nvcc_args.extend(
                    [
                        "--device-entity-has-hidden-visibility=false",
                        "-static-global-template-stub=false",
                    ]
                )
        if cub_home is None:
            prefix = os.environ.get("CONDA_PREFIX", None)
            if prefix is not None and os.path.isdir(prefix + "/include/cub"):
                cub_home = prefix + "/include"

        if cub_home is None:
            warnings.warn(
                "The environment variable `CUB_HOME` was not found. "
                "NVIDIA CUB is required for compilation and can be downloaded "
                "from `https://github.com/NVIDIA/cub/releases`. You can unpack "
                "it to a location of your choice and set the environment variable "
                "`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
            )
        else:
            include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            nvcc_args.extend(nvcc_flags_env.split(" "))

        # This is needed for pytorch 1.6 and earlier. See e.g.
        # https://github.com/facebookresearch/pytorch3d/issues/436
        # It is harmless after https://github.com/pytorch/pytorch/pull/47404 .
        # But it can be problematic in torch 1.7.0 and 1.7.1
        if torch.__version__[:4] != "1.7.":
            CC = os.environ.get("CC", None)
            if CC is not None:
                existing_CC = get_existing_ccbin(nvcc_args)
                if existing_CC is None:
                    CC_arg = "-ccbin={}".format(CC)
                    nvcc_args.append(CC_arg)
                elif existing_CC != CC:
                    msg = f"Inconsistent ccbins: {CC} and {existing_CC}"
                    raise ValueError(msg)

        extra_compile_args["nvcc"] = nvcc_args

    sources = [os.path.relpath(s, this_dir) for s in sources]

    ext_modules = [
        extension(
            "pytorch3d._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, use_ninja=False, **kwargs)

else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension


setup(
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
