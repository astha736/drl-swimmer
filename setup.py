"""Package configuration for the DRL swimmer project."""

from pathlib import Path
import sys


if __name__ == "__main__" and len(sys.argv) == 1:
    print("No setup.py command supplied.")
    print("Install this project with:")
    print("  python -m pip install -e . --no-build-isolation")
    print("or run the full project installer:")
    print("  bash setup.sh")
    sys.exit(0)

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from farms_core import get_include_paths
from setuptools import Extension, find_packages, setup


PROJECT_ROOT = Path(__file__).parent
DEBUG = False

Options.docstrings = True
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False
Options.clear_to_none = True
Options.annotate = False
Options.fast_fail = False
Options.warning_errors = False
Options.error_on_unknown_names = True
Options.error_on_uninitialized = True
Options.convert_range = True
Options.cache_builtins = True
Options.gcc_branch_hints = True
Options.lookup_module_cpdef = False
Options.embed = None
Options.cimport_from_pyx = False
Options.buffer_max_dims = 8
Options.closure_freelist_size = 8


def build_extensions():
    """Build optional AgnathaX Cython extensions."""
    sources = sorted(Path("agnathax_control").glob("*.pyx"))
    if not sources:
        return []

    extensions = [
        Extension(
            ".".join(source.with_suffix("").parts),
            sources=[source.as_posix()],
            extra_compile_args=["-O3"],
            extra_link_args=["-O3"],
        )
        for source in sources
    ]
    return cythonize(
        extensions,
        include_path=[np.get_include()] + get_include_paths(),
        compiler_directives={
            "binding": False,
            "embedsignature": True,
            "cdivision": True,
            "language_level": 3,
            "infer_types": True,
            "profile": True,
            "wraparound": False,
            "boundscheck": DEBUG,
            "nonecheck": DEBUG,
            "initializedcheck": DEBUG,
            "overflowcheck": DEBUG,
            "overflowcheck.fold": DEBUG,
            "cdivision_warnings": DEBUG,
            "always_allow_keywords": DEBUG,
            "linetrace": DEBUG,
            "optimize.use_switch": True,
            "optimize.unpack_method_calls": True,
            "warn.undeclared": True,
            "warn.unreachable": True,
            "warn.maybe_uninitialized": True,
            "warn.unused": True,
            "warn.unused_arg": True,
            "warn.unused_result": True,
            "warn.multiple_declarators": True,
        },
    )


setup(
    name="drl-swimmer",
    version="0.1.0",
    author="Astha Gupta",
    license="MIT",
    packages=find_packages(exclude=("env_drl_swimmer", "env_drl_swimmer.*")),
    include_package_data=True,
    package_data={"agnathax_control": ["*.pxd"]},
    include_dirs=[np.get_include()],
    ext_modules=build_extensions(),
    zip_safe=False,
    python_requires=">=3.10",
)
