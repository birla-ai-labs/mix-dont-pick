"""Generator registry for coverage-to-utility study."""

from .arima import ARIMASynthesizer
from .chaotic import ChaoticSynthesizer
from .ets import ETSSynthesizer
from .fbm import FBMSynthesizer
from .garch import GARCHSynthesizer
from .kernelsynth import KernelSynthesizer
from .sde import SDESynthesizer
from .stepfunction import StepFunctionSynthesizer
from .timesynth_gen import TimeSynthesizer
from .tsi import TSISynthesizer
from .waveform import WaveformSynthesizer

GENERATOR_REGISTRY = {
    "kernelsynth": KernelSynthesizer,
    "garch": GARCHSynthesizer,
    "timesynth": TimeSynthesizer,
    "tsi": TSISynthesizer,
    "sde": SDESynthesizer,
    "stepfunction": StepFunctionSynthesizer,
    "arima": ARIMASynthesizer,
    "fbm": FBMSynthesizer,
    "ets": ETSSynthesizer,
    "waveform": WaveformSynthesizer,
    "chaotic": ChaoticSynthesizer,
}

__all__ = [
    "GENERATOR_REGISTRY",
    "ARIMASynthesizer",
    "ChaoticSynthesizer",
    "ETSSynthesizer",
    "FBMSynthesizer",
    "GARCHSynthesizer",
    "KernelSynthesizer",
    "SDESynthesizer",
    "StepFunctionSynthesizer",
    "TimeSynthesizer",
    "TSISynthesizer",
    "WaveformSynthesizer",
]