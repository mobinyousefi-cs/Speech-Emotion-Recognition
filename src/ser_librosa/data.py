#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Dataset loading and preprocessing utilities for RAVDESS-based Speech Emotion Recognition.

Responsibilities:
- Walk the RAVDESS speech directory and collect `.wav` file paths.
- Parse emotion codes from file names and map them to human-readable labels.
- Filter by selected emotions and remove under-represented classes.
- Extract features using `features.extract_features_from_file`.
- Produce train/test splits ready for modeling.

Usage:
from ser_librosa.data import load_ravdess_dataset

Notes:
- Configuration (paths, selected emotions, etc.) is defined in config.py.
=================================================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .config import (
    EMOTION_CODE_TO_LABEL,
    MIN_SAMPLES_PER_CLASS,
    RANDOM_STATE,
    RAVDESS_SPEECH_DIR,
    SELECTED_EMOTIONS,
    TEST_SIZE,
)
from .features import extract_features_from_file


def _parse_emotion_from_filename(path: Path) -> str | None:
    """Parse the emotion label from a RAVDESS-style filename.

    Example filename: `03-01-06-01-02-01-12.wav`
    Split by '-', index 2 is the emotion code (here '06').

    Returns
    -------
    str or None
        Parsed emotion label, or None if it cannot be parsed.
    """

    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return EMOTION_CODE_TO_LABEL.get(emotion_code)


def _collect_files(root: Path) -> List[Path]:
    """Collect all `.wav` files recursively from a root directory."""

    return sorted(root.rglob("*.wav"))


def load_ravdess_dataset(
    data_root: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Load the RAVDESS speech dataset and return train/test splits.

    Parameters
    ----------
    data_root:
        Optional path to the RAVDESS speech directory. If None, uses
        `config.RAVDESS_SPEECH_DIR`.

    Returns
    -------
    X_train, X_test, y_train, y_test, label_encoder
        Feature matrices and label arrays, plus the fitted LabelEncoder.
    """

    root = data_root or RAVDESS_SPEECH_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"RAVDESS speech directory not found: {root}. "
            "Please update config.RAVDESS_SPEECH_DIR or pass data_root explicitly."
        )

    files = _collect_files(root)
    if not files:
        raise RuntimeError(f"No .wav files found under {root}")

    print(f"Found {len(files)} audio files under {root}")

    features_list: List[np.ndarray] = []
    labels: List[str] = []

    for wav_path in tqdm(files, desc="Extracting features", unit="file"):
        emotion = _parse_emotion_from_filename(wav_path)
        if emotion is None:
            continue
        if emotion not in SELECTED_EMOTIONS:
            continue

        try:
            feats = extract_features_from_file(wav_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to process {wav_path}: {exc}")
            continue

        features_list.append(feats)
        labels.append(emotion)

    if not features_list:
        raise RuntimeError("No features extracted. Check dataset paths and filters.")

    X = np.stack(features_list, axis=0)
    y = np.array(labels)

    # Filter out classes with too few samples
    class_counts: Dict[str, int] = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1

    keep_mask = np.array([class_counts[label] >= MIN_SAMPLES_PER_CLASS for label in y])
    X = X[keep_mask]
    y = y[keep_mask]

    print("Class counts (after MIN_SAMPLES_PER_CLASS filter):")
    for label, count in sorted(class_counts.items()):
        if count >= MIN_SAMPLES_PER_CLASS:
            print(f"  {label:10s}: {count}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    return X_train, X_test, y_train, y_test, le
