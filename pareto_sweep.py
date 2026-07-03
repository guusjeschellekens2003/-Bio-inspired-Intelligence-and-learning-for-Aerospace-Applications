import numpy as np
import json5
import matplotlib.pyplot as plt
from evaluation import single_evaluate
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import Problem

#set evaluation to "baseline"
# Fixed seeds for deterministic training landscapes across the front
TEST_SEEDS = [40, 41, 42, 43, 44]


def run_weight_sweep():
    with open("config.json") as f:
        config = json5.load(f)

    config["cell_size"] = (2 * config["world_size"]) / config["grid_size"]
    config["n_inputs"] = 2 * config["n_agents"] + 12
    config["n_outputs"] = 2
    config["n_param"] = (config["n_hidden"] * config["n_inputs"] +
                         config["n_hidden"] +
                         config["n_hidden"] * config["n_outputs"] +
                         config["n_outputs"])


    weight_sets = [
        (3.0, 0.1, "Pure exploration"),
        (2.5, 0.5, "Explore-biased"),
        (1.5, 1.5, "Balanced"),
        (0.5, 2.5, "Connect-biased"),
        (0.1, 3.0, "Pure connectivity"),
    ]


    class WeightedProblem(Problem):
        def __init__(self, w_cov, w_con, cfg):
            self.w_cov = w_cov
            self.w_con = w_con
            self.cfg = cfg
            super().__init__(n_var=cfg["n_param"], n_obj=1, n_constr=0,
                             xl=-100 * np.ones(cfg["n_param"]),
                             xu=+100 * np.ones(cfg["n_param"]))

        def _evaluate(self, x, out, *args, **kwargs):
            pop_size = x.shape[0]
            total_fits = np.zeros(pop_size)

            for seed in TEST_SEEDS:
                _, cov, con, eng, col, obs = single_evaluate(
                    self.cfg, x, return_components=True, terrain_seed=seed
                )
                total_fits += (self.w_cov * (1.0 - cov)) + \
                              (self.w_con * (1.0 - con)) + \
                              (0.3 * eng) + \
                              (3.0 * col) + \
                              (3.0 * obs)

            out["F"] = (total_fits / len(TEST_SEEDS)).reshape(-1)

    results = []

    for w_cov, w_con, label in weight_sets:
        print(f"\n--- Training Profile: {label} ---")


        problem = WeightedProblem(w_cov, w_con, config)

        res = minimize(
            problem,
            CMAES(x0=np.random.randn(config["n_param"]) * 0.1, sigma=0.2, popsize=config["n_genomes"]),
            DefaultSingleObjectiveTermination(ftol=1e-6, n_max_gen=config.get("n_gen", 200)),
            verbose=True
        )

        # Evaluate best genome 20 times over random environments for clean statistics
        best = res.X.reshape(1, -1)
        covs, cons = [], []
        for seed_idx in range(20):
            fit, cov, con, nrg, col, obs = single_evaluate(config, best, return_components=True, terrain_seed=seed_idx)
            covs.append(float(cov[0]))
            cons.append(float(con[0]))

        mean_cov = np.mean(covs)
        mean_con = np.mean(cons)
        results.append((mean_cov, mean_con, label))

        np.save(f"genome_{label.replace(' ', '_')}.npy", res.X)
        print(f"-> Optimized {label} Result -> Cov: {mean_cov * 100:.1f}% | Conn: {mean_con * 100:.1f}%")


    fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure for bigger text


    results.sort(key=lambda item: item[0])


    ax.plot([r[0] * 100 for r in results], [r[1] * 100 for r in results],
            color='darkorange', linestyle='--', alpha=0.8, linewidth=2.5, zorder=4)


    for cov, con, label in results:
        ax.scatter(cov * 100, con * 100, color='teal', s=200, zorder=5, edgecolor='black')

        # Shift the labels slightly so they don't overlap the dots
        ax.annotate(label, (cov * 100, con * 100),
                    textcoords="offset points", xytext=(12, -5),
                    fontsize=14, weight='bold')


    ax.set_xlabel("Area Coverage (%)", fontsize=20, weight='bold', labelpad=15)
    ax.set_ylabel("Uplink Connectivity (%)", fontsize=20, weight='bold', labelpad=15)
    ax.set_title("Coverage vs. Connectivity Trade-off\n(Pareto Front via Objective Weight Sweeping)",
                 fontsize=24, pad=20)

    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)


    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=150)
    print("\nSaved pareto_front.png successfully.")


if __name__ == "__main__":
    run_weight_sweep()