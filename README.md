# OpenExo Project AI4

Research project for modular perception and controller experiments using MyoAssist and MuJoCo.

The project separates perception from control so that different perception methods and controllers can later be combined and evaluated using the same interface.

## Project structure

- `src/perception/` - perception modules
- `src/controllers/` - controller modules
- `src/environment/` - MyoAssist / MuJoCo environment
- `src/common/` - shared interfaces and data structures
- `src/evaluation/` - evaluation utilities
- `tests/` - basic integration and simulation tests


## Architecture

The intended pipeline is:

    MyoAssist environment
            |
            v
      Raw observation
            |
            v
    Perception module
            |
            v
     PerceptionState
            |
            v
     Controller module
            |
            v
      Exoskeleton action

The shared interface between perception and control is `PerceptionState` in `src/common/types.py`.

## Environment setup

Requirements:

- Python 3.12
- Git
- uv

Create a virtual environment:

    uv venv --python 3.12 .venv

Activate it on macOS / Linux:

    source .venv/bin/activate

Activate it on Windows PowerShell:

    .venv\Scripts\Activate.ps1

Install the project:

    uv pip install -e .

MyoAssist is installed directly from the pinned Git commit defined in `pyproject.toml`.

Current MyoAssist commit:

    b1baf69da59dc6728515d0977cbc8fbf00b9225a

The project applies the dependency overrides required by the current MyoAssist setup:

- MuJoCo 3.4.x
- dm-control 1.0.36

## Tests

Test the perception-controller pipeline:

    python -m tests.test_pipeline

Test that the MyoAssist model loads:

    python -m tests.test_openexo

Run the MuJoCo simulation smoke test:

    python -m tests.test_simulation

The simulation test performs multiple MuJoCo steps and checks that the model state remains finite.

## Development

Perception implementations belong in:

    src/perception/

Controller implementations belong in:

    src/controllers/

Shared interfaces belong in:

    src/common/

Environment-specific code belongs in:

    src/environment/

Local virtual environments, generated results, checkpoints, IDE files, packaging metadata, and Python cache files should not be committed.

## Current status

The environment setup, modular perception-controller interface, MyoAssist model loading, and basic MuJoCo simulation stepping have been verified.

The exact exoskeleton model configuration used for the final experiments will be verified separately before running the full perception-controller experiment matrix.