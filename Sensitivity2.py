import numpy as np
import json5
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import Problem
from evaluation import single_evaluate


N_AGENTS_LIST = [2, 5, 8, 11, 14]

# Fixed benchmark seeds to prevent noisy evaluation landscapes during retraining
BENCHMARK_SEEDS = [40,41,42,43,44]


def optimize_swarm_size(n_agents, base_config):
    config = base_config.copy()
    config["n_agents"] = n_agents
    config["comm_range_leader"] = 1.5  # Standard baseline unit value

    # Dynamically re-map neural layouts for this specific swarm capacity
    config["cell_size"] = (2 * config["world_size"]) / config["grid_size"]
    config["n_inputs"] = 2 * config["n_agents"] + 12
    config["n_outputs"] = 2
    config["n_param"] = (config["n_hidden"] * config["n_inputs"] +
                         config["n_hidden"] +
                         config["n_hidden"] * config["n_outputs"] +
                         config["n_outputs"])

    class DynamicSizeProblem(Problem):
        def __init__(self):
            super().__init__(n_var=config["n_param"], n_obj=1, n_constr=0,
                             xl=-100 * np.ones(config["n_param"]), xu=+100 * np.ones(config["n_param"]))

        def _evaluate(self, x, out, *args, **kwargs):
            total_fits = np.zeros(x.shape[0])
            for seed in BENCHMARK_SEEDS:
                # Unpacks correctly using your updated 6-value signature
                res = single_evaluate(config, x, return_components=True, terrain_seed=seed)
                total_fits += res[0]  # res[0] is still raw fitness
            out["F"] = (total_fits / len(BENCHMARK_SEEDS)).reshape(-1)

    res = minimize(
        DynamicSizeProblem(),
        CMAES(x0=np.random.randn(config["n_param"]) * 0.1, sigma=0.2, popsize=config["n_genomes"]),
        DefaultSingleObjectiveTermination(ftol=1e-5, n_max_gen=100),
        verbose=True
    )


    best_genome = res.X.reshape(1, -1)
    run_covs, run_cons, run_cols = [], [], []

    for seed in BENCHMARK_SEEDS:

        fit, cov, con, nrg, col, obs = single_evaluate(config, best_genome, return_components=True, terrain_seed=seed)
        run_covs.append(cov[0] * 100)
        run_cons.append(con[0] * 100)
        run_cols.append(col[0] * 100)  # Track collision percentage trends

    return np.mean(run_covs), np.mean(run_cons), np.mean(run_cols)


def main():
    with open("config.json") as f:
        base_config = json5.load(f)

    final_covs, final_cons, final_cols = [], [], []

    print("Starting Sweep B: Agent Scaling Capacity (With Retraining)...")
    for agents in N_AGENTS_LIST:
        print(f"Optimizing swarm size N={agents}...")
        mean_cov, mean_con, mean_col = optimize_swarm_size(agents, base_config)
        final_covs.append(mean_cov)
        final_cons.append(mean_con)
        final_cols.append(mean_col)
        print(f"   Results -> Coverage: {mean_cov:.1f}% | Connectivity: {mean_con:.1f}% | Collisions: {mean_col:.2f}%")


    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Primary Axis: Mission Success Metrics
    ax1.plot(N_AGENTS_LIST, final_covs, '-o', label='Area Coverage (%)', color='teal', linewidth=2)
    ax1.plot(N_AGENTS_LIST, final_cons, '-s', label='Uplink Connectivity (%)', color='darkorange', linewidth=2)
    ax1.set_xlabel('Swarm Size (Total Agents)')
    ax1.set_ylabel('Performance Metrics (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle=':', alpha=0.6)


    ax2 = ax1.twinx()
    ax2.plot(N_AGENTS_LIST, final_cols, '--x', label='Pairwise Collision Rate (%)', color='crimson', alpha=0.7)
    ax2.set_ylabel('Collision Frequency (%)', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_transform=fig.transFigure)

    plt.title('Sweep B: Swarm Scalability Capacity vs Crowding Trade-Offs')
    fig.tight_layout()
    plt.savefig('sweep_b_agents.png', dpi=150)
    print("Saved sweep_b_agents.png successfully.")


if __name__ == "__main__":
    main()