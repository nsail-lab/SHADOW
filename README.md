# Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games

Official codebase for the paper:

**Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games**
Accepted at AAMAS 2026 (International Conference on Autonomous Agents and Multiagent Systems), AAAI Track.

This repository contains all code necessary to:

* Train agents from scratch in the proposed pursuit–evasion environment
* Reproduce the experimental setup described in the paper
* Evaluate trained models
* Analyze training dynamics and performance metrics

---

## Overview

The paper studies strategic communication under risk in a pursuit–evasion setting, where agents learn to balance information exchange with potential threat exposure.

The codebase includes:

* The full environment implementation
* Reinforcement learning training pipeline
* Evaluation scripts
* Notebooks for metrics visualization and analysis

---

## Repository Structure

Main files:

* `environment.yml` – Conda environment specification
* `main.py` – Main entry point to train models from scratch
* `test_eval.ipynb` – Notebook to evaluate trained models (includes working example)
* `train_eval.ipynb` – Notebook to visualize and analyze training metrics

---

## Installation

We recommend using Conda to replicate the exact software environment used for the experiments.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate UUV_DG
```

This installs all required dependencies with the correct versions.

---

## Training from Scratch

To train agents from scratch, run:

```bash
python main.py [arguments]
```

All core experimental parameters can be controlled via command-line arguments.

### Configurable Parameters

You can configure:

**Agent configuration**

* Type of evader
* Type of pursuer
* Opponent modeling
* Memory length

**Environment parameters**

* Agent speeds
* Acceleration limits
* Elimination radius
* Other environment dynamics

**Training hyperparameters**

* Number of training steps
* Discount factor
* Learning rate(s)
* Exploration parameters
* Other RL-related settings

Refer directly to `main.py` for the complete list of available arguments and default values.

---

## Observation Space Dimensions

The observation space dimensions are defined inside `main.py` as:

```python
evader_state_dim = 19
pursuer_state_dim = 19
```

If you modify the observation/state representation, you must manually update these values accordingly.

---

## Evaluating Trained Models

To evaluate a trained model, use:

```
test_eval.ipynb
```

This notebook:

* Loads a trained model checkpoint
* Runs evaluation episodes
* Computes performance metrics
* Includes a working example

You can adapt it to evaluate different saved models or configurations.

---

## Analyzing Training Dynamics

The notebook:

```
train_eval.ipynb
```

provides visualizations and analysis tools for training behavior across episodes, including:

* Pursuer reward evolution
* Evader reward evolution
* Communication-related metrics
* Other experiment-specific statistics

These analyses were used to generate plots and results reported in the paper.

---

## Reproducibility Guidelines

To reproduce results from the paper:

1. Create the environment using `environment.yml`
2. Run training with `main.py` using the parameter settings described in the paper
3. Use `train_eval.ipynb` to inspect learning curves
4. Use `test_eval.ipynb` to evaluate final performance

For improved reproducibility:

* Fix random seeds
* Use the dependency versions specified in `environment.yml`
* Keep hardware configuration consistent when possible

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lagatta2025strategiccommunicationthreatlearning,
      title={Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games}, 
      author={Valerio La Gatta and Dolev Mutzari and Sarit Kraus and VS Subrahmanian},
      year={2025},
      eprint={2510.07813},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.07813}, 
}
```

---
