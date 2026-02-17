# Robust and Safe Locomotion under Actuator Failures using Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Stable%20Baselines3-brightgreen)](https://stable-baselines3.readthedocs.io/)
[![Simulator](https://img.shields.io/badge/Simulator-MuJoCo-orange)](https://mujoco.org/)
[![Read Report](https://img.shields.io/badge/Read-Full%20Report-red?style=flat&logo=adobe-acrobat-reader)](docs/Robust_Locomotion_DRL_Paper.pdf)

The project investigates robustness and safety in deep reinforcement learning for legged locomotion under actuator-level failures.  
The objective is to evaluate whether structured exposure to variability during training enables policies to maintain stable behavior under unseen mechanical degradations and domain shifts.

Experiments are conducted on MuJoCo benchmark environments:

- **Ant** (redundant quadruped morphology)  
- **Hopper** (underactuated biped morphology)  

The learning algorithm used in all experiments is **Soft Actor-Critic (SAC)** implemented with Stable-Baselines3.

---

## Research Motivation

Policies trained under nominal actuation often exhibit abrupt performance degradation when exposed to actuator faults or dynamic uncertainty.

This project studies whether robustness and safety can emerge from:

- actuator-level domain randomization  
- goal-conditioned task formulation  
- safety-aware reward shaping  
- terrain friction randomization  

Importantly, the learning algorithm remains unchanged across experiments.  
Robustness mechanisms are introduced exclusively at the environment and reward level.

---

## Methodology Overview

The framework integrates the following components:

### Actuator-Level Domain Randomization
- Stochastic torque scaling during training  
- Deterministic actuator faults during evaluation  
- Evaluation under structured fault scenarios  

### Goal-Conditioned Locomotion
- Spatial target-reaching tasks  
- Success-based evaluation metrics  

### Safety-Aware Reward Shaping
- Penalization of torso instability  
- Penalization of excessive vertical velocity  
- Stability-oriented task design  

### Terrain Domain Randomization
- Friction scaling  
- Optional curriculum strategy  

All robustness mechanisms are implemented via modular environment wrappers.

---

## Repository Structure

### `ant/`
Goal-conditioned locomotion with actuator and terrain randomization.

Main scripts:
- `train_sac_ant_resilient.py`
- `train_sac_ant_goal_safe.py`
- `custom_ant.py`

### `hopper/`
Forward locomotion under actuator degradation.

Main scripts:
- `run_experiments.py`
- `configs.py`
- `custom_hopper.py`

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/robust-locomotion-drl.git
cd robust-locomotion-drl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

MuJoCo must be correctly installed and configured.

---

## Usage

### Ant experiments

```bash
cd ant
python train_sac_ant_resilient.py
```

### Hopper experiments

```bash
cd hopper
python run_experiments.py
```

Evaluation scripts generate performance metrics under nominal and faulted conditions.

---

## Evaluation Metrics

The following metrics are reported:

* Mean episode return
* Success rate (goal-conditioned tasks)
* Mean time-to-goal
* Safety violation rate

Stress-test evaluations compare baseline and resilient policies under deterministic actuator degradation.

---

## Project Report

A detailed project report is available in:

```
docs/Robust_Locomotion_DRL_Paper.pdf
```

The report includes:

* theoretical background
* methodological design
* experimental protocol
* quantitative results and analysis

```

