#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: test_features.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Basic unit tests for the feature extraction module.

These tests validate that:
- Feature extraction runs without error for a short synthetic signal.
- The resulting feature vector has a stable, positive length.

Usage:
pytest tests/test_features.py
=================================================================================================================
"""

from __future__ import annotations

import numpy as np

from ser_librosa.config import SAMPLE_RATE
from ser_librosa.features import extract_features


def test_extract_features_shape() -> None:
    """Feature extraction should return a 1D vector with positive length."""

    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    # Simple sinusoid
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    feats = extract_features(y, SAMPLE_RATE)

    assert feats.ndim == 1
    assert feats.size > 0
    # No NaNs or infs
    assert np.all(np.isfinite(feats))
