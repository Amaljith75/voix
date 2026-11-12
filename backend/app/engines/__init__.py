"""Engine adapters. Importing this package registers every built-in adapter.

The RVC and Qwen3-TTS adapters import PyTorch lazily, so registering them is
cheap; they report themselves unavailable when the AI runtime is missing.
"""
from . import piper_engine  # noqa: F401
from . import qwen3_engine  # noqa: F401
from . import rvc  # noqa: F401
