#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Speech Emotion Recognition with librosa
File: test_model_training.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-07
Updated: 2025-11-07
License: MIT License (see LICENSE file for details)
=
Description:
Lightweight sanity tests for the model training utilities.

These tests use a small synthetic dataset to verify that:
- The default MLPClassifier can fit and achieve > 0.9 accuracy on an easy task.

Usage:
pytest tests/test_model_training.py
=================================================================================================================
"""

from __future__ import annotations

import numpy as np

from ser_librosa.model import build_default_mlp, train_and_evaluate


def test_mlp_can_fit_simple_dataset() -> None:
    """MLP should overfit a simple linearly separable dataset."""

    rng = np.random.default_rng(0)

    # Two Gaussian blobs in 2D
    n_per_class = 50
    mean0 = np.array([-1.0, -1.0])
    mean1 = np.array([1.0, 1.0])
    cov = np.eye(2) * 0.1

    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    model = build_default_mlp()
    # Shorten training for the test
    model.max_iter = 200

    fitted, train_acc, test_acc = train_and_evaluate(
        model=model,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        target_names=("class0", "class1"),
    )

    assert train_acc > 0.9
    assert test_acc > 0.9
    assert fitted is not None
