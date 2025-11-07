#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Central configuration for the Speech Emotion Recognition project.

This file defines experiment-wide constants such as:
- Paths (data root, RAVDESS location, default model output directory)
- Random seeds
- Emotion mappings and selected target emotions
- Feature extraction parameters
- Model hyperparameters

Usage:
from ser_librosa.config import DATA_ROOT, RAVDESS_SPEECH_DIR

Notes:
- Adjust these values to run different experiments.
- Paths are defined relative to the repository root by default.
=================================================================================================================
"""

from __future__ import annotations

from pathlib import Path

# -------------------------------------------------------------------------------------------------
# General settings
# -------------------------------------------------------------------------------------------------

# Reproducibility
RANDOM_STATE: int = 42

# Project root is assumed to be the directory containing `pyproject.toml`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data directories
DATA_ROOT: Path = PROJECT_ROOT / "data" / "raw"
RAVDESS_SPEECH_DIR: Path = DATA_ROOT / "RAVDESS_SPEECH"

# Models directory
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Emotion mapping (RAVDESS)
# -------------------------------------------------------------------------------------------------

# RAVDESS filename pattern: MM-VC-EE-II-SS-RR-AAA.wav
# Where EE is the emotion code (2nd index if split by '-')
EMOTION_CODE_TO_LABEL = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# You can restrict experiments to a subset of emotions if desired.
SELECTED_EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# -------------------------------------------------------------------------------------------------
# Feature extraction parameters (librosa)
# -------------------------------------------------------------------------------------------------

SAMPLE_RATE: int = 22050
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MFCC: int = 40

# Whether to include extra spectral features in addition to MFCCs.
USE_CHROMA: bool = True
USE_MEL: bool = True
USE_CONTRAST: bool = True
USE_TONNETZ: bool = True

# -------------------------------------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------------------------------------

# Train/test split
TEST_SIZE: float = 0.2

# Minimum number of samples per emotion to be kept (for safety)
MIN_SAMPLES_PER_CLASS: int = 5

# -------------------------------------------------------------------------------------------------
# MLP hyperparameters
# -------------------------------------------------------------------------------------------------

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 32,
    "learning_rate": "adaptive",
    "max_iter": 500,
    "shuffle": True,
    "random_state": RANDOM_STATE,
    "early_stopping": True,
    "validation_fraction": 0.1,
}
