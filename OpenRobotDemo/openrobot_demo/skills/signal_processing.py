"""Signal processing skills: filtering, FFT, feature extraction.

These skills operate on sensor data streams and produce processed signals
that can be consumed by downstream skills or the world model.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class LowPassFilterSkill(SkillInterface):
    """Apply a low-pass filter to a 1D signal."""

    name = "low_pass_filter"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Apply a Butterworth low-pass filter to a 1D signal (e.g. IMU data, force readings).",
            parameters=[
                ParamSchema(name="data", type="list", description="1D signal array.", required=True),
                ParamSchema(name="sample_rate", type="float", description="Sampling rate in Hz.", required=True),
                ParamSchema(name="cutoff_hz", type="float", description="Cutoff frequency in Hz.", required=True),
                ParamSchema(name="order", type="int", description="Filter order (default 4).", required=False, default=4),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether filtering succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="filtered", type="list", description="Filtered signal array."),
            ],
            dependencies=[],
        )

    def execute(self, data: List[float], sample_rate: float, cutoff_hz: float, order: int = 4, **kwargs) -> Dict[str, Any]:
        try:
            arr = np.array(data, dtype=np.float64)
            if len(arr) < order * 2 + 1:
                return {"success": False, "message": f"Data too short ({len(arr)} samples) for order-{order} filter."}
            nyq = sample_rate / 2.0
            normal_cutoff = cutoff_hz / nyq
            b, a = scipy_signal.butter(order, normal_cutoff, btype="low", analog=False)
            filtered = scipy_signal.filtfilt(b, a, arr)
            return {"success": True, "message": f"Low-pass filtered at {cutoff_hz}Hz.", "filtered": filtered.tolist()}
        except Exception as exc:
            logger.exception("[LowPassFilterSkill] Failed")
            return {"success": False, "message": str(exc)}


class KalmanFilter1DSkill(SkillInterface):
    """1D Kalman filter for state estimation from noisy measurements."""

    name = "kalman_filter_1d"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Apply a 1D Kalman filter to estimate true state from noisy measurements (position, distance, etc.).",
            parameters=[
                ParamSchema(name="measurements", type="list", description="Noisy measurement sequence.", required=True),
                ParamSchema(name="process_noise", type="float", description="Process noise covariance Q (default 0.01).", required=False, default=0.01),
                ParamSchema(name="measurement_noise", type="float", description="Measurement noise covariance R (default 1.0).", required=False, default=1.0),
                ParamSchema(name="initial_estimate", type="float", description="Initial state estimate.", required=False, default=0.0),
                ParamSchema(name="initial_error", type="float", description="Initial estimate error covariance.", required=False, default=1.0),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether filtering succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="estimates", type="list", description="Filtered state estimates."),
            ],
            dependencies=[],
        )

    def execute(self, measurements: List[float], process_noise: float = 0.01,
                measurement_noise: float = 1.0, initial_estimate: float = 0.0,
                initial_error: float = 1.0, **kwargs) -> Dict[str, Any]:
        try:
            z = np.array(measurements, dtype=np.float64)
            n = len(z)
            estimates = np.zeros(n)

            x = initial_estimate  # state estimate
            P = initial_error     # error covariance
            Q = process_noise
            R = measurement_noise

            for i in range(n):
                # Predict
                x_pred = x
                P_pred = P + Q
                # Update
                K = P_pred / (P_pred + R)
                x = x_pred + K * (z[i] - x_pred)
                P = (1 - K) * P_pred
                estimates[i] = x

            return {
                "success": True,
                "message": f"Kalman filter applied to {n} measurements.",
                "estimates": estimates.tolist(),
            }
        except Exception as exc:
            logger.exception("[KalmanFilter1DSkill] Failed")
            return {"success": False, "message": str(exc)}


class FFTSkill(SkillInterface):
    """Compute FFT of a signal for frequency analysis."""

    name = "fft_analysis"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Compute FFT of a signal and return dominant frequencies (e.g. vibration analysis).",
            parameters=[
                ParamSchema(name="data", type="list", description="1D signal array.", required=True),
                ParamSchema(name="sample_rate", type="float", description="Sampling rate in Hz.", required=True),
                ParamSchema(name="top_k", type="int", description="Number of dominant frequencies to return.", required=False, default=3),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether FFT succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="frequencies", type="list", description="Frequency bins (Hz)."),
                ResultSchema(name="amplitudes", type="list", description="Amplitude spectrum."),
                ResultSchema(name="dominant", type="list", description="Top-k dominant frequencies [(freq, amp), ...]."),
            ],
            dependencies=[],
        )

    def execute(self, data: List[float], sample_rate: float, top_k: int = 3, **kwargs) -> Dict[str, Any]:
        try:
            arr = np.array(data, dtype=np.float64)
            n = len(arr)
            if n < 2:
                return {"success": False, "message": "Data too short for FFT."}

            freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
            amps = np.abs(np.fft.rfft(arr))

            # Find dominant frequencies
            idx = np.argsort(amps)[::-1][:top_k]
            dominant = [(float(freqs[i]), float(amps[i])) for i in idx]

            return {
                "success": True,
                "message": f"FFT computed, {len(freqs)} frequency bins.",
                "frequencies": freqs.tolist(),
                "amplitudes": amps.tolist(),
                "dominant": dominant,
            }
        except Exception as exc:
            logger.exception("[FFTSkill] Failed")
            return {"success": False, "message": str(exc)}
