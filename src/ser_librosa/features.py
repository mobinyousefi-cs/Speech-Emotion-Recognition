#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: features.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Feature extraction utilities for Speech Emotion Recognition.

This module uses librosa to compute a rich set of time-averaged audio features:
- MFCCs
- Chroma features
- Mel-scaled spectrogram
- Spectral contrast
- Tonnetz features (from harmonic component)

Usage:
from ser_librosa.features import extract_features_from_file

Notes:
- All features are aggregated by taking the mean over time, resulting in a fixed-length vector.
- Configure feature parameters in config.py.
=================================================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np

from .config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MFCC,
    USE_CHROMA,
    USE_MEL,
    USE_CONTRAST,
    USE_TONNETZ,
)


def _safe_load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load an audio file as a floating point time series.

    Parameters
    ----------
    path:
        Path to the audio file.

    Returns
    -------
    y, sr:
        Audio time series and sample rate.
    """

    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    # Ensure we have non-empty audio
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract a fixed-length feature vector from an audio signal.

    Parameters
    ----------
    y:
        Audio time series.
    sr:
        Sample rate.

    Returns
    -------
    np.ndarray
        1D feature vector.
    """

    features: Iterable[np.ndarray] = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfccs_mean = np.mean(mfccs, axis=1)
    features = [mfccs_mean]

    # Chroma
    if USE_CHROMA:
        stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.append(np.mean(chroma, axis=1))

    # Mel-scaled spectrogram
    if USE_MEL:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(np.mean(mel, axis=1))

    # Spectral contrast
    if USE_CONTRAST:
        stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.append(np.mean(contrast, axis=1))

    # Tonnetz (on harmonic component)
    if USE_TONNETZ:
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        features.append(np.mean(tonnetz, axis=1))

    return np.concatenate(features, axis=0)


def extract_features_from_file(path: Path) -> np.ndarray:
    """Convenience wrapper: load audio and compute features.

    Parameters
    ----------
    path:
        Path to a `.wav` file.

    Returns
    -------
    np.ndarray
        Feature vector for the file.
    """

    y, sr = _safe_load_audio(path)
    return extract_features(y, sr)
