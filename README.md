# Robust Locomotion via Curriculum Domain Randomization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Stable%20Baselines3-brightgreen)](https://stable-baselines3.readthedocs.io/)
[![Simulator](https://img.shields.io/badge/Simulator-MuJoCo-orange)](https://mujoco.org/)
[![Read Report](https://img.shields.io/badge/Read-Full%20Report-red?style=flat&logo=adobe-acrobat-reader)](docs/Robust_Locomotion_DRL_Paper.pdf)

This repository hosts a comprehensive **Deep Reinforcement Learning (DRL)** framework for resilient legged locomotion. It investigates **Sim-to-Sim transfer** and **Actuator Fault Tolerance** on two distinct robotic platforms: the **Quadrupedal Ant** and the **Bipedal Hopper**.

Using **Curriculum Domain Randomization (UDR)** and **Safety-Aware Rewards**, the agents learn to navigate dynamic environments and recover from severe mechanical failures (e.g., 50% power loss).

---

## 📂 Project Structure

The repository is organized into two main modules based on the robot type:

### 1. 🐜 `ant/` (Quadrupedal Navigation)
Focuses on **Goal-Conditioned Navigation** under dynamic variations.
* **Core Task:** Reach dynamic $(x, y)$ targets while handling heavier legs and slippery terrain.
* **Key Scripts:**
    * `train_sac_ant_resilient.py`: Main training loop with Actuator UDR and Fault Injection.
    * `train_sac_ant_goal_safe.py`: Training with Goal-Safe wrapper and friction curriculum.
    * `custom_ant.py`: Custom environment definition with dynamic randomization capabilities.

### 2. 🦘 `hopper/` (Bipedal Resilience)
Focuses on **Survival & Balance** under critical actuator faults using a modular config system.
* **Core Task:** Maintain forward velocity while respecting safety constraints (pitch/velocity).
* **Key Scripts:**
    * `run_experiments.py`: Unified entry point for training and evaluation.
    * `configs.py`: Configuration definitions for faults, rewards, and safety specs.
    * `custom_hopper.py`: Hopper environment with truncated normal distribution randomization.

---

## 🚀 Key Features

* **Curriculum Domain Randomization:** The difficulty of the environment (friction, mass, motor gain) increases progressively during training, stabilizing the learning curve.
* **Actuator Fault Tolerance:** Agents are trained to withstand significant motor degradation. The framework includes a wrapper that simulates torque loss at specific timesteps during evaluation.
* **Safety-Aware Control:**
    * **Ant:** Uses `GoalSafeWrapper` to penalize dangerous tilts and strictly define success conditions.
    * **Hopper:** Uses `SafetyConstraintWrapper` to enforce limits on torso pitch and vertical speed.
* **Robust Evaluation:** A comprehensive evaluation pipeline generates "Stress-Test Grids," plotting Success Rate vs. Fault Strength to quantify resilience.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/robust-locomotion-drl.git](https://github.com/YOUR_USERNAME/robust-locomotion-drl.git)
    cd robust-locomotion-drl
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

### Running Ant Experiments
To train the Ant agent with Resilient Motor UDR:
```bash
cd ant
python train_sac_ant_resilient.py

```

*This will save models to `ant/models/` and logs to `ant/logs/`.*

### Running Hopper Experiments

The Hopper module uses a unified runner. To start the interactive menu:

```bash
cd hopper
python run_experiments.py

```

*Select option `(1)` for Training or `(3)` for Full Pipeline (Train + Eval).*

### Plotting Results

To generate comparison plots (Success Rate vs Fault Strength) for Hopper:

```bash
cd hopper
python plot_results.py

```

## 📄 Project Report

A detailed PDF report is available in the `docs/` folder. It contains:

* **Methodology:** Explanation of the Domain Randomization and Curriculum approach.
* **Math:** The exact formulas used for the Rewards and Safety constraints.
* **Results:** Graphs and charts comparing the Resilient agents against the Baselines.

👉 **[Read the Report (PDF)](docs/Robust_Locomotion_DRL_Paper.pdf)**


