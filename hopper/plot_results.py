from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")
IN_CSV = RUNS_DIR / "results.csv"

OUT_DIR = RUNS_DIR / "plots_eval_trends"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _read_results() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing results CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Normalize numeric columns we rely on
    numeric_cols = [
        "mean_return",
        "std_return",
        "mean_ep_len",
        "success_rate",
        "fault_strength",
        "fault_start_step",
        "seed",
        "total_timesteps",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str)
    if "exp" in df.columns:
        df["exp"] = df["exp"].astype(str)

    return df


def _savefig(name: str) -> None:
    plt.gcf().set_constrained_layout(True)
    plt.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()

def _sort_fault_df(dd: pd.DataFrame) -> pd.DataFrame:
    """
    Sort fault cases in a stable and readable way:
    (actuator index, strength, start step, tag)
    """
    if dd.empty:
        return dd

    out = dd.copy()

    if "fault_indices" not in out.columns:
        out["fault_indices"] = ""
    if "fault_strength" not in out.columns:
        out["fault_strength"] = np.nan
    if "fault_start_step" not in out.columns:
        out["fault_start_step"] = np.nan
    if "fault_tag" not in out.columns:
        out["fault_tag"] = ""

    out["_idx"] = out["fault_indices"].astype(str)
    out["_s"] = pd.to_numeric(out["fault_strength"], errors="coerce").fillna(999.0)
    out["_t0"] = pd.to_numeric(out["fault_start_step"], errors="coerce").fillna(999.0)
    out["_tag"] = out["fault_tag"].astype(str)

    out = out.sort_values(
        by=["_idx", "_s", "_t0", "_tag"],
        kind="mergesort",
    ).reset_index(drop=True)

    return out.drop(columns=["_idx", "_s", "_t0", "_tag"], errors="ignore")

def _aggregate_fault_mode(d_fault: pd.DataFrame):
    """
    Aggregate across fault cases into ONE (mean, std).
    std_total = sqrt( E[sigma_i^2] + Var(mu_i) )
    """
    if d_fault.empty:
        return np.nan, np.nan

    mu_i = d_fault["mean_return"].to_numpy(dtype=float)
    sigma_i = d_fault["std_return"].fillna(0.0).to_numpy(dtype=float)

    mu = float(np.nanmean(mu_i))
    within = float(np.nanmean(sigma_i ** 2))
    between = float(np.nanvar(mu_i))
    std_total = float(np.sqrt(max(0.0, within + between)))

    return mu, std_total

def plot_eval_trends(df: pd.DataFrame) -> None:
    """
    For EACH (exp, seed) generate:
      A) eval vs eval_fault
      B) eval_goal vs eval_fault_goal
      C) eval, eval_fault, eval_goal, eval_fault_goal (aggregated)
    """

    exps = sorted(df["exp"].unique())
    seeds = sorted(df["seed"].dropna().unique())

    for exp in exps:
        for seed in seeds:
            d = df[(df["exp"] == exp) & (df["seed"] == seed)]
            if d.empty:
                continue

            def _get(mode):
                r = d[d["mode"] == mode]
                if r.empty:
                    return np.nan, np.nan
                mu = float(r["mean_return"].iloc[0])
                std = float(r["std_return"].iloc[0]) if "std_return" in r.columns else 0.0
                return mu, std

            eval_mu, eval_std = _get("eval")
            goal_mu, goal_std = _get("eval_goal")

            d_fault = _sort_fault_df(d[d["mode"] == "eval_fault"])
            d_fault_goal = _sort_fault_df(d[d["mode"] == "eval_fault_goal"])

            if not d_fault.empty and np.isfinite(eval_mu):
                x = np.arange(len(d_fault))
                y = d_fault["mean_return"].to_numpy()
                s = d_fault["std_return"].fillna(0.0).to_numpy()

                plt.figure()
                plt.plot(x, y, marker="o", label="eval_fault")
                plt.fill_between(x, y - s, y + s, alpha=0.2)

                plt.plot([x[0], x[-1]], [eval_mu, eval_mu], "--", label="eval baseline")
                plt.xlabel("Fault case index")
                plt.ylabel("Mean return")
                plt.title(f"{exp} | seed {int(seed)} — eval vs eval_fault")
                plt.legend()
                _savefig(f"{exp}_seed{int(seed)}_A_eval_vs_eval_fault")

            if not d_fault_goal.empty and np.isfinite(goal_mu):
                x = np.arange(len(d_fault_goal))
                y = d_fault_goal["mean_return"].to_numpy()
                s = d_fault_goal["std_return"].fillna(0.0).to_numpy()

                plt.figure()
                plt.plot(x, y, marker="o", label="eval_fault_goal")
                plt.fill_between(x, y - s, y + s, alpha=0.2)

                plt.plot([x[0], x[-1]], [goal_mu, goal_mu], "--", label="eval_goal baseline")
                plt.xlabel("Fault case index")
                plt.ylabel("Mean return")
                plt.title(f"{exp} | seed {int(seed)} — eval_goal vs eval_fault_goal")
                plt.legend()
                _savefig(f"{exp}_seed{int(seed)}_B_eval_goal_vs_eval_fault_goal")

            fault_mu, fault_std = _aggregate_fault_mode(d_fault)
            fault_goal_mu, fault_goal_std = _aggregate_fault_mode(d_fault_goal)

            modes = ["eval", "eval_fault", "eval_goal", "eval_fault_goal"]
            y = np.array([eval_mu, fault_mu, goal_mu, fault_goal_mu], dtype=float)
            s = np.array([eval_std, fault_std, goal_std, fault_goal_std], dtype=float)

            if np.isfinite(y).sum() >= 2:
                x = np.arange(len(modes))

                plt.figure()
                plt.plot(x, y, marker="o")
                plt.fill_between(x, y - s, y + s, alpha=0.2)
                plt.xticks(x, modes, rotation=15)
                plt.ylabel("Mean return")
                plt.title(f"{exp} | seed {int(seed)} — all eval modes")
                _savefig(f"{exp}_seed{int(seed)}_C_all_modes")

    print(f"[PLOTS] A/B/C saved in: {OUT_DIR}")

def plot_eval_modes_xaxis(df: pd.DataFrame) -> None:
    """
    Single plot (one figure) for ALL experiments:
      - X axis is ordered evaluation modes: eval -> eval_fault -> eval_goal -> eval_fault_goal
      - One line per (exp, seed)
      - Fault modes are aggregated across fault cases (mean over rows)
    """

    EVAL_ORDER = ["eval", "eval_fault", "eval_goal", "eval_fault_goal"]
    exps = sorted(df["exp"].unique())
    seeds = sorted(df["seed"].dropna().unique())

    plt.figure()

    for exp in exps:
        for seed in seeds:
            d = df[(df["exp"] == exp) & (df["seed"] == seed)]
            if d.empty:
                continue

            y = []
            for mode in EVAL_ORDER:
                r = d[d["mode"] == mode]
                if r.empty:
                    y.append(np.nan)
                else:
                    # For fault modes there are many rows (fault cases) -> mean across them
                    y.append(float(r["mean_return"].mean()))

            x = np.arange(len(EVAL_ORDER), dtype=int)

            plt.plot(
                x,
                np.array(y, dtype=float),
                marker="o",
                linewidth=2,
                label=f"{exp} | seed={int(seed)}",
            )

    plt.xticks(np.arange(len(EVAL_ORDER)), EVAL_ORDER, rotation=15)
    plt.ylabel("Mean return")
    plt.title("Mean return across eval modes (all experiments & seeds)")
    plt.legend(ncol=2, fontsize=8)
    _savefig("ALL_EXPS_D_eval_modes_per_seed")

def main():
    df = _read_results()
    plot_eval_trends(df)
    plot_eval_modes_xaxis(df)


if __name__ == "__main__":
    main()
