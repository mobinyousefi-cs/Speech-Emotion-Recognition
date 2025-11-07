#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Package initialization for the Speech Emotion Recognition (SER) project.

This module exposes a small, clean public API for higher-level usage.

Usage:
from ser_librosa import load_dataset, build_default_model

Notes:
- See config.py for experiment-wide configuration.
- See train.py and infer.py for end-to-end CLI scripts.
=================================================================================================================
"""

from .config import (
    DATA_ROOT,
    RAVDESS_SPEECH_DIR,
    RANDOM_STATE,
    SELECTED_EMOTIONS,
)
from .data import load_ravdess_dataset
from .model import build_default_mlp

__all__ = [
    "DATA_ROOT",
    "RAVDESS_SPEECH_DIR",
    "RANDOM_STATE",
    "SELECTED_EMOTIONS",
    "load_ravdess_dataset",
    "build_default_mlp",
]
