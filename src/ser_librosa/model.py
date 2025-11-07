#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Model utilities for Speech Emotion Recognition using scikit-learn.

This module defines:
- `build_default_mlp` for constructing an MLPClassifier with sensible defaults.
- `train_and_evaluate` for training the model and printing evaluation metrics.

Usage:
from ser_librosa.model import build_default_mlp, train_and_evaluate

Notes:
- Hyperparameters are configured in config.MLP_PARAMS.
=================================================================================================================
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from .config import MLP_PARAMS

console = Console()


def build_default_mlp() -> MLPClassifier:
    """Construct the default MLPClassifier for SER.

    Returns
    -------
    sklearn.neural_network.MLPClassifier
        Unfitted classifier instance.
    """

    return MLPClassifier(**MLP_PARAMS)


def train_and_evaluate(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Tuple[str, ...],
) -> Tuple[MLPClassifier, float, float]:
    """Train the given model and evaluate it on train and test sets.

    Parameters
    ----------
    model:
        Unfitted MLPClassifier.
    X_train, y_train:
        Training data.
    X_test, y_test:
        Test data.
    target_names:
        Tuple of label names in the order of their encoded indices.

    Returns
    -------
    model, train_acc, test_acc
        Fitted model and accuracy scores.
    """

    console.rule("[bold blue]Training MLPClassifier")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    table = Table(title="Accuracy")
    table.add_column("Split", style="cyan", justify="center")
    table.add_column("Accuracy", style="magenta", justify="center")
    table.add_row("Train", f"{train_acc:.4f}")
    table.add_row("Test", f"{test_acc:.4f}")
    console.print(table)

    console.rule("[bold green]Classification Report (Test)")
    report = classification_report(y_test, y_pred_test, target_names=target_names)
    console.print(report)

    return model, train_acc, test_acc
