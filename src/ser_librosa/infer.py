#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: infer.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Command-line script for running inference with a trained Speech Emotion Recognition model.

Given a path to a `.wav` file and a trained model artifact (joblib), this script:
- Extracts audio features using librosa.
- Loads the trained MLP model and label encoder.
- Predicts the most likely emotion (and optionally prints probabilities).

Usage:
python -m ser_librosa.infer --model-path models/mlp_ser.joblib --audio-path path/to/file.wav --show-probs

Notes:
- Ensure the model artifact was created using train.py and matches the feature pipeline.
=================================================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from rich.console import Console
from rich.table import Table

from .features import extract_features_from_file

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run inference with a trained Speech Emotion Recognition model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained model artifact (joblib)",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        required=True,
        help="Path to the input audio file (.wav)",
    )
    parser.add_argument(
        "--show-probs",
        action="store_true",
        help="Whether to print class probabilities for all emotions.",
    )

    return parser.parse_args()


def _load_artifact(path: Path) -> Dict[str, object]:
    """Load the trained model artifact from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    required_keys = {"model", "label_encoder"}
    if not required_keys.issubset(artifact.keys()):
        raise ValueError(
            f"Artifact at {path} does not contain required keys: {sorted(required_keys)}",
        )
    return artifact


def main() -> None:
    """Run SER inference for a single audio file."""

    args = parse_args()

    console.rule("[bold blue]Loading model artifact")
    artifact = _load_artifact(args.model_path)
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]

    console.print(f"Loaded model from [bold]{args.model_path}[/bold]")

    console.rule("[bold blue]Extracting features")
    features = extract_features_from_file(args.audio_path)
    X = np.expand_dims(features, axis=0)

    console.rule("[bold blue]Predicting emotion")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    predicted_label = label_encoder.inverse_transform(y_pred)[0]
    console.print(f"Predicted emotion: [bold green]{predicted_label}[/bold green]")

    if args.show_probs:
        classes = label_encoder.inverse_transform(np.arange(y_proba.shape[1]))
        probs = y_proba[0]

        table = Table(title="Class probabilities")
        table.add_column("Emotion", style="cyan")
        table.add_column("Probability", style="magenta", justify="right")

        for cls, p in sorted(zip(classes, probs), key=lambda cp: cp[1], reverse=True):
            table.add_row(cls, f"{p:.4f}")

        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
