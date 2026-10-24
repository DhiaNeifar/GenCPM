"""Builds the C++ late-fusion core (gencpm.evaluation.fusion_core). All metadata is in pyproject.toml."""
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

# -ffp-contract=off keeps IEEE semantics (no fused multiply-add), so results do not depend on the compiler
ext = Pybind11Extension("gencpm.evaluation.fusion_core", ["gencpm/evaluation/csrc/fusion_core.cpp"],
                        cxx_std=17, extra_compile_args=["-O3", "-ffp-contract=off"])
setup(ext_modules=[ext])
