"""Voice activity detection for microphone audio streamed to the server.

The server needs to know, frame by frame, whether the user is talking: to time the silence that
ends a turn, to find the pauses where it checks whether the question is complete, and to notice
the user talking over an answer (barge-in).

* ``EnergyVAD`` (default, no model): short-term energy against an adaptive noise floor. Works
  well on the noise-suppressed, echo-cancelled audio browsers deliver.
* ``SileroVAD``: the Silero neural VAD via sherpa-onnx, when VAD_MODEL points at silero_vad.onnx.
  More robust to background noise and music.

Both take 16 kHz float frames of any length and report whether the latest audio is speech.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from app.voice.audio import STT_RATE, rms_dbfs

logger = logging.getLogger(__name__)


class VAD(Protocol):
    name: str

    def is_speech(self, samples: np.ndarray) -> bool: ...


class EnergyVAD:
    name = "energy"

    def __init__(self, threshold_db: float = 12.0, min_level_db: float = -50.0, floor_db: float = -65.0):
        self.threshold_db = threshold_db   # speech is this far above the noise floor ...
        self.min_level_db = min_level_db   # ... and at least this loud
        self.floor = floor_db

    def is_speech(self, samples: np.ndarray) -> bool:
        level = rms_dbfs(samples)
        speech = level > max(self.floor + self.threshold_db, self.min_level_db)
        if level < self.floor:
            self.floor = 0.7 * self.floor + 0.3 * level       # follow the noise floor down quickly
        elif not speech:
            self.floor = 0.98 * self.floor + 0.02 * level     # and up slowly, only in non-speech
        return speech


class SileroVAD:
    name = "silero"

    def __init__(self, model: str, threshold: float = 0.5):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = 0.05  # its hangover delays every end of turn
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.sample_rate = STT_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

    def is_speech(self, samples: np.ndarray) -> bool:
        self._vad.accept_waveform(samples)
        while not self._vad.empty():     # segments are not needed: drop them to bound memory
            self._vad.pop()
        return bool(self._vad.is_speech_detected())


class VADFactory:
    """One detector per voice connection (they keep state); the model file is checked once."""

    def __init__(self, model: str = ""):
        self.model = model if model and Path(model).is_file() else ""
        if model and not self.model:
            logger.warning("VAD_MODEL %s not found; using the energy detector", model)
        if self.model:
            try:
                SileroVAD(self.model)
            except Exception as exc:  # sherpa-onnx missing or model unreadable
                logger.warning("Silero VAD unavailable (%s); using the energy detector", exc)
                self.model = ""

    @property
    def name(self) -> str:
        return "silero" if self.model else "energy"

    def __call__(self) -> VAD:
        return SileroVAD(self.model) if self.model else EnergyVAD()
