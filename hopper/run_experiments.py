# project_extension/extension/run_experiments.py
"""
This module provides a unified entry point for running training, evaluation,
and result visualization of the reinforcement learning experiments defined
in the project.

It orchestrates the execution of multiple experimental configurations and
random seeds by leveraging the shared experiment registry (`EXPERIMENTS`)
and reproducibility setup (`SEEDS`). The module supports four execution modes:
training only, evaluation only, combined training and evaluation, and result
plotting.

Training is performed using vectorized environments to improve sample
efficiency, while evaluation is executed in deterministic mode to ensure
consistent and comparable performance metrics across runs.

The script is designed to be executed as a standalone command-line interface,
allowing the user to interactively select the desired execution mode.
This structure promotes modularity, reproducibility, and scalability of
experimental workflows.
"""
from __future__ import annotations

from .configs import EXPERIMENTS, SEEDS
from .train import train_one
from .evaluate import evaluate_one
from .plot_results import main as plot_main


def run_training(n_envs: int = 4) -> None:
    print("=== TRAINING ===")
    for cfg in EXPERIMENTS:
        for seed in SEEDS:
            print(f"[TRAIN] exp={cfg.name} seed={seed}")
            train_one(cfg, seed=seed, n_envs=n_envs)


def run_evaluation() -> None:
    print("=== EVALUATION ===")
    for cfg in EXPERIMENTS:
        for seed in SEEDS:
            evaluate_one(cfg, seed=seed, deterministic=True)


def run_training_and_evaluation(n_envs: int = 4) -> None:
    run_training(n_envs=n_envs)
    run_evaluation()


def main() -> None:
    print("\nVuoi eseguire:\n  (1) Training\n  (2) Evaluation\n  (3) Entrambi\n  (4) Plot")
    choice = input("> ").strip()

    if choice == "1":
        run_training(n_envs=4)
    elif choice == "2":
        run_evaluation()
    elif choice == "3":
        run_training_and_evaluation(n_envs=4)
    elif choice == "4":
        plot_main()
    else:
        print("Scelta non valida.")


if __name__ == "__main__":
    main()
