#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Command-line entry point for training the Speech Emotion Recognition model.

Steps:
1. Load the RAVDESS dataset (features + labels).
2. Split into train/test sets.
3. Build an MLPClassifier.
4. Train the model and print evaluation metrics.
5. Save the trained model and label encoder with joblib.

Usage:
python -m ser_librosa.train --data-root data/raw/RAVDESS_SPEECH --model-out models/mlp_ser.joblib

Notes:
- Configure experiment behavior in config.py.
=================================================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from rich.console import Console

from .config import MODELS_DIR, RAVDESS_SPEECH_DIR
from .data import load_ravdess_dataset
from .model import build_default_mlp, train_and_evaluate

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train a Speech Emotion Recognition model on RAVDESS using librosa + MLP."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=RAVDESS_SPEECH_DIR,
        help="Path to RAVDESS speech directory (default: config.RAVDESS_SPEECH_DIR)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=MODELS_DIR / "mlp_ser.joblib",
        help="Path to save the trained model (joblib)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training routine."""

    args = parse_args()

    console.rule("[bold blue]Loading dataset")
    X_train, X_test, y_train, y_test, label_encoder = load_ravdess_dataset(args.data_root)

    target_names = tuple(label_encoder.inverse_transform(sorted(set(y_train))))

    console.rule("[bold blue]Building model")
    model = build_default_mlp()

    model, train_acc, test_acc = train_and_evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_names=target_names,
    )

    # Save model and label encoder
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "target_names": target_names,
    }
    joblib.dump(artifact, args.model_out)

    console.rule("[bold green]Training complete")
    console.print(f"Model saved to [bold]{args.model_out}[/bold]")


if __name__ == "__main__":  # pragma: no cover
    main()
