"""
SpikeGate — A modular framework for dynamic hardware gating in Spiking Transformers.

Provides tools for profiling, gating, ablation, head importance estimation,
and comprehensive report generation on arbitrary pretrained spiking models.
"""

from ._version import __version__
from .converter import convert_to_gated_snn, SpikingModelWrapper
from .gating import DynamicGateController
from .attention import SpikingGatedAttention
from .profiler import AutoProfiler
from .evaluator import GatingAblationStudy
from .neurons import LIFNode
from .policies import GatingPolicy
from .reporter import ReportGenerator

__all__ = [
    "__version__",
    "convert_to_gated_snn",
    "SpikingModelWrapper",
    "DynamicGateController",
    "SpikingGatedAttention",
    "AutoProfiler",
    "GatingAblationStudy",
    "LIFNode",
    "GatingPolicy",
    "ReportGenerator",
]
