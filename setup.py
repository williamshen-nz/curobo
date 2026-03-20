#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""curobo package setuptools."""

# NOTE: This file is still needed to allow the package to be
# installed in editable mode.
#
# References:
# * https://setuptools.pypa.io/en/latest/setuptools.html#setup-cfg-only-projects

# Standard Library
import hashlib
import importlib.machinery
import sys
from pathlib import Path

# Third Party
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_LIB_DIR = Path("src/curobo/curobolib")
_FINGERPRINT_FILE = _LIB_DIR / ".build_fingerprint"



def _source_fingerprint() -> str:
    """SHA-256 of all CUDA/C++ source contents — content-based, not mtime-based."""
    src_files = sorted((_LIB_DIR / "cpp").glob("*.cu")) + sorted((_LIB_DIR / "cpp").glob("*.cpp"))
    h = hashlib.sha256()
    for f in src_files:
        h.update(f.read_bytes())
    return h.hexdigest()


def _so_files_up_to_date(ext_names: list[str]) -> bool:
    """Return True if compiled .so files exist and match the current source fingerprint."""
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]  # e.g. .cpython-312-x86_64-linux-gnu.so
    so_files = [_LIB_DIR / f"{name}{ext_suffix}" for name in ext_names]
    if not all(f.exists() for f in so_files):
        return False
    if not _FINGERPRINT_FILE.exists():
        return False
    return _FINGERPRINT_FILE.read_text().strip() == _source_fingerprint()


class BuildExtensionWithFingerprint(BuildExtension):
    """Manages a source fingerprint file for CUDA build cache invalidation.

    Deletes the fingerprint before building and rewrites it after success, so
    an interrupted build always invalidates the cache on the next run.
    """

    def run(self):
        _FINGERPRINT_FILE.unlink(missing_ok=True)
        super().run()
        _FINGERPRINT_FILE.write_text(_source_fingerprint() + "\n")


extra_cuda_args = {
    "nvcc": [
        "--threads=8",
        "-O3",
        "--ftz=true",
        "--fmad=true",
        "--prec-div=false",
        "--prec-sqrt=false",
    ]
}

if sys.platform == "win32":
    extra_cuda_args["nvcc"].append("--allow-unsupported-compiler")

_ext_names = ["lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu"]

if _so_files_up_to_date(_ext_names):
    pass  # fast path: pip install just re-registers the editable install, no CUDA build
    ext_modules = []
else:
    ext_modules = [
        CUDAExtension(
            "curobo.curobolib.lbfgs_step_cu",
            [
                "src/curobo/curobolib/cpp/lbfgs_step_cuda.cpp",
                "src/curobo/curobolib/cpp/lbfgs_step_kernel.cu",
            ],
            extra_compile_args=extra_cuda_args,
        ),
        CUDAExtension(
            "curobo.curobolib.kinematics_fused_cu",
            [
                "src/curobo/curobolib/cpp/kinematics_fused_cuda.cpp",
                "src/curobo/curobolib/cpp/kinematics_fused_kernel.cu",
            ],
            extra_compile_args=extra_cuda_args,
        ),
        CUDAExtension(
            "curobo.curobolib.line_search_cu",
            [
                "src/curobo/curobolib/cpp/line_search_cuda.cpp",
                "src/curobo/curobolib/cpp/line_search_kernel.cu",
                "src/curobo/curobolib/cpp/update_best_kernel.cu",
            ],
            extra_compile_args=extra_cuda_args,
        ),
        CUDAExtension(
            "curobo.curobolib.tensor_step_cu",
            [
                "src/curobo/curobolib/cpp/tensor_step_cuda.cpp",
                "src/curobo/curobolib/cpp/tensor_step_kernel.cu",
            ],
            extra_compile_args=extra_cuda_args,
        ),
        CUDAExtension(
            "curobo.curobolib.geom_cu",
            [
                "src/curobo/curobolib/cpp/geom_cuda.cpp",
                "src/curobo/curobolib/cpp/sphere_obb_kernel.cu",
                "src/curobo/curobolib/cpp/pose_distance_kernel.cu",
                "src/curobo/curobolib/cpp/self_collision_kernel.cu",
            ],
            extra_compile_args=extra_cuda_args,
        ),
    ]

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionWithFingerprint},
    package_data={"": ["*.so"]},
    include_package_data=True,
)
