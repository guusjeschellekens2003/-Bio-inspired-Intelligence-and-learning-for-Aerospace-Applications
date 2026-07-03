import numpy as np
import json5
import matplotlib.pyplot as plt
from evaluation import single_evaluate

# Metric ranges in meters converted to simulation units
METERS_RANGES = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
SIM_COMM_RANGES = [m / 25.0 for m in METERS_RANGES]
NUM_SEEDS = 30


def main():
    with open("config.json") as f:
        config = json5.load(f)

    config["n_agents"] = 5
    config["cell_size"] = (2 * config["world_size"]) / config["grid_size"]
    config["n_inputs"] = 2 * config["n_agents"] + 12
    config["n_outputs"] = 2

    genome = np.load("best_genome_new2.npy").reshape(1, -1)

    mean_coverages, std_coverages = [], []
    mean_connectivities, std_connectivities = [], []

    print("Starting Sweep A: Communication Range Sensitivity...")
    for m_val, sim_val in zip(METERS_RANGES, SIM_COMM_RANGES):
        config["comm_range_leader"] = sim_val
        run_covs, run_cons = [], []

        for seed in range(NUM_SEEDS):
            _, cov, con, _, _, _ = single_evaluate(config, genome, return_components=True, terrain_seed=seed)
            run_covs.append(cov[0] * 100)
            run_cons.append(con[0] * 100)

        mean_coverages.append(np.mean(run_covs))
        std_coverages.append(np.std(run_covs))
        mean_connectivities.append(np.mean(run_cons))
        std_connectivities.append(np.std(run_cons))


    fig, ax1 = plt.subplots(figsize=(10, 6))


    ax1.errorbar(METERS_RANGES, mean_coverages, yerr=std_coverages, fmt='-o', color='teal',
                 label='Area Coverage (%)', markersize=8, capsize=5, linewidth=2)
    ax1.set_xlabel('Leader Communication Range (m)', fontsize=20, labelpad=10)
    ax1.set_ylabel('Area Coverage (%)', color='teal', fontsize=20, labelpad=10)
    ax1.tick_params(axis='y', labelcolor='teal', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)


    ax2 = ax1.twinx()
    ax2.errorbar(METERS_RANGES, mean_connectivities, yerr=std_connectivities, fmt='-s', color='darkorange',
                 label='Uplink Connectivity (%)', markersize=8, capsize=5, linewidth=2)
    ax2.set_ylabel('Uplink Connectivity (%)', color='darkorange', fontsize=20, labelpad=10)
    ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=14)

    plt.title('Communication Range Sensitivity Assessment', fontsize=24, pad=15)

    fig.tight_layout()
    plt.savefig('sweep_a_communication.png', dpi=150)
    print("Saved sweep_a_communication.png successfully.")


if __name__ == "__main__":
    main()