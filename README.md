# Speech Emotion Recognition with librosa

> **Complete Python mini-project with src/ layout, tests, CI, and clean architecture.**

This repository implements a **Speech Emotion Recognition (SER)** pipeline using

- **librosa** and **soundfile** for audio loading and feature extraction
- **scikit-learn** for building an **MLPClassifier**
- The **RAVDESS** speech subset as the reference dataset

The project is structured to match a **professional, research-friendly workflow** and is ready to be published on GitHub under your profile.

---

## 1. Project Overview

**Goal:** Train and evaluate a model that can recognize **emotion from speech** using handcrafted acoustic features (MFCCs, chroma, spectral features) extracted via **librosa**, and an **MLP classifier** from scikit-learn.

**Target emotions** (RAVDESS speech subset typically includes):

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

You can easily change the selected emotions in `src/ser_librosa/config.py`.

---

## 2. Repository Structure

```text
speech-emotion-recognition-librosa/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ .gitignore
├─ .editorconfig
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ src/
│  └─ ser_librosa/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ features.py
│     ├─ data.py
│     ├─ model.py
│     ├─ train.py
│     └─ infer.py
└─ tests/
   ├─ test_features.py
   └─ test_model_training.py
```

- **`src/ser_librosa/config.py`** – Central configuration (paths, emotions, feature parameters, model hyperparameters).
- **`src/ser_librosa/features.py`** – Feature extraction utilities (MFCCs, chroma, mel-spectrogram, spectral contrast, tonnetz, etc.).
- **`src/ser_librosa/data.py`** – Dataset loading for RAVDESS; file parsing, label mapping, and train/test split.
- **`src/ser_librosa/model.py`** – Model factory and train/evaluate helpers.
- **`src/ser_librosa/train.py`** – Command-line training script that trains the model and saves it to disk.
- **`src/ser_librosa/infer.py`** – Simple CLI to run inference on a single `.wav` file using a trained model.
- **`tests/`** – Minimal unit tests to verify features and a lightweight training loop.

---

## 3. Dataset Setup (RAVDESS)

This project assumes you are using the **RAVDESS speech** subset.

1. Download the preprocessed RAVDESS archive (as given in your reference link) and extract it locally.
2. Place the **speech files** under:

```text
<data_root>/RAVDESS_SPEECH/
    Actor_01/
        03-01-01-01-01-01-01.wav
        ...
    Actor_02/
        ...
    ...
```

3. In `src/ser_librosa/config.py`, set:

```python
DATA_ROOT = Path("data/raw")
RAVDESS_SPEECH_DIR = DATA_ROOT / "RAVDESS_SPEECH"
```

Then create the directory on disk and move your extracted RAVDESS speech folders there:

```bash
mkdir -p data/raw
# Move or copy the extracted speech folders into data/raw/RAVDESS_SPEECH
```

The **file name convention** in RAVDESS (e.g. `03-01-06-01-02-01-12.wav`) embeds the emotion code:

- Position 3 (when split on `-`) is the **emotion ID**
- Mapping (used here by default):

  | Code | Emotion   |
  |------|-----------|
  | 01   | neutral   |
  | 02   | calm      |
  | 03   | happy     |
  | 04   | sad       |
  | 05   | angry     |
  | 06   | fearful   |
  | 07   | disgust   |
  | 08   | surprised |

The loader in `data.py` parses these codes and converts them into string labels.

---

## 4. Installation & Environment

We use **`pyproject.toml`** (PEP 621) as the single source of truth for dependencies.

### 4.1. Create and activate a virtual environment

```bash
python -m venv .venv
# On Linux/Mac
source .venv/bin/activate
# On Windows
.venv\\Scripts\\activate
```

### 4.2. Install the project in editable mode

```bash
pip install --upgrade pip
pip install .[dev]
```

This will install:

- Core runtime deps: `librosa`, `soundfile`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `tqdm`, `rich`.
- Dev deps (via `[dev]` extra): `pytest`, `black`, `ruff`.

---

## 5. Training the Model

Once the dataset is prepared and dependencies are installed, run:

```bash
python -m ser_librosa.train \
  --data-root data/raw/RAVDESS_SPEECH \
  --model-out models/mlp_ser.joblib
```

**What this does:**

1. Recursively scans the RAVDESS speech directory for `.wav` files.
2. Extracts features for each file using `features.extract_features_from_file`.
3. Encodes labels and splits the dataset into train/test sets (stratified).
4. Builds an `MLPClassifier` with hyperparameters from `config.py`.
5. Trains the model and prints:
   - Train accuracy
   - Test accuracy
   - Classification report
6. Saves the fitted model and label encoder with `joblib` into the `models/` directory.

You can adjust:

- **Features** in `features.py`
- **Model hyperparameters** in `config.py`
- **Test size and random seeds** in `config.py`

---

## 6. Running Inference on a New Audio File

After training, run inference on a single `.wav` file:

```bash
python -m ser_librosa.infer \
  --model-path models/mlp_ser.joblib \
  --audio-path path/to/example.wav
```

Output example:

```text
Loaded model from models/mlp_ser.joblib
Predicted emotion: angry
Predicted probabilities:
  neutral   : 0.01
  calm      : 0.02
  happy     : 0.05
  sad       : 0.10
  angry     : 0.70
  fearful   : 0.08
  disgust   : 0.02
  surprised : 0.02
```

If you pass `--show-probs`, the script prints a probability distribution for all classes.

---

## 7. Configuration

Key configuration values live in `src/ser_librosa/config.py`:

- **Random seeds** and deterministic behavior
- **Paths**: data root, default model output path
- **Emotion mapping** and selected target emotions
- **Feature extraction parameters**: number of MFCCs, FFT window size, hop length, etc.
- **Model hyperparameters**: hidden layer sizes, activation, learning rate, regularization, max iterations

You can fully reconfigure the experiment by editing this single file.

---

## 8. Running Tests & Linting

We provide a minimal but useful CI setup.

### 8.1. Run tests

```bash
pytest
```

### 8.2. Lint with Ruff

```bash
ruff check src tests
```

### 8.3. Check formatting with Black

```bash
black --check src tests
```

The GitHub Actions workflow `.github/workflows/ci.yml` runs all three (lint, format-check, tests) on every push and pull request.

---

## 9. Suggested Experiments

- **Change features**: use only MFCCs vs. MFCC + chroma + spectral contrast; compare accuracy.
- **Vary test size**: 10%, 20%, 30%; observe stability.
- **Tune MLP architecture**: number of layers, hidden units, activation functions.
- **Try other models**: SVM, RandomForest, GradientBoosting using the same features.
- **Cross-validation**: extend `model.py` to run k-fold cross-validation instead of single train/test split.

Document your results in a markdown report or Jupyter notebook in a `notebooks/` folder if you wish to extend the project.

---

## 10. License & Attribution

- This project is released under the **MIT License** (see `LICENSE`).
- Dataset: **RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song**.
  Please follow the original dataset license and citation instructions when using it in publications.

---

## 11. Quick Start TL;DR

```bash
# 1. Clone your repo and enter it
# git clone https://github.com/mobinyousefi-cs/speech-emotion-recognition-librosa.git
# cd speech-emotion-recognition-librosa

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install .[dev]

# 3. Prepare dataset
mkdir -p data/raw
# (Extract RAVDESS speech subset into data/raw/RAVDESS_SPEECH)

# 4. Train model
python -m ser_librosa.train --data-root data/raw/RAVDESS_SPEECH --model-out models/mlp_ser.joblib

# 5. Inference on a single file
python -m ser_librosa.infer --model-path models/mlp_ser.joblib --audio-path path/to/example.wav --show-probs
```

You now have a clean, professional **Speech Emotion Recognition** project ready for GitHub and for future research extensions.

