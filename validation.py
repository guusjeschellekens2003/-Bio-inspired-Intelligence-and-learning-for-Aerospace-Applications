import numpy as np
import json5
from evaluation import single_evaluate


def validate_genome(genome_path, config_path, ablation_type="baseline"):
    with open(config_path) as f:
        config = json5.load(f)
    with open("config.json") as f:
        config = json5.load(f)
    config["cell_size"]      = (2 * config["world_size"]) / config["grid_size"]


    # Re-calculate correct cell size
    config["cell_size"] = (2 * config["world_size"]) / config["grid_size"]

    # Adjust n_inputs explicitly based on which file we are testing
    if ablation_type == "baseline":
        config["n_inputs"] = 2 * config["n_agents"] + 12
        config["n_outputs"] = 2
        config["n_param"] = (
                config["n_hidden"] * config["n_inputs"] +
                config["n_hidden"] +
                config["n_hidden"] * config["n_outputs"] +
                config["n_outputs"]
        )

    elif ablation_type == "no_lvd" or ablation_type == "no_target_dir":
        config["n_inputs"] = 2 * config["n_agents"] + 10
        config["n_outputs"] = 2
        config["n_param"] = (
                config["n_hidden"] * config["n_inputs"] +
                config["n_hidden"] +
                config["n_hidden"] * config["n_outputs"] +
                config["n_outputs"]
        )
    elif ablation_type == "no_time":
        config["n_inputs"] = 2 * config["n_agents"] + 11
        config["n_outputs"] = 2
        config["n_param"] = (
                config["n_hidden"] * config["n_inputs"] +
                config["n_hidden"] +
                config["n_hidden"] * config["n_outputs"] +
                config["n_outputs"]
        )

    elif ablation_type == "no_rel_pos":
        config["n_inputs"] = 14
        config["n_outputs"] = 2
        config["n_param"] = (
                config["n_hidden"] * config["n_inputs"] +
                config["n_hidden"] +
                config["n_hidden"] * config["n_outputs"] +
                config["n_outputs"]
        )

    # Load the specific genome we want to test
    # Reshape it to mock a population of size 1 for single_evaluate
    genome = np.load(genome_path).reshape(1, -1)

    coverages = []
    connectivities = []

    # Evaluate across 30 completely different map seeds
    num_test_maps = 30
    for seed in range(num_test_maps):

        fitness, coverage, connect, _, _ = single_evaluate(
            config, genome, analysis=True, terrain_seed=seed
        )

        coverages.append(coverage[0] * 100)  # Convert to %
        connectivities.append(connect[0] * 100)  # Convert to %

    print(f"=== Results for {genome_path} ===")
    print(f"Coverage:     {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}%")
    print(f"Connectivity: {np.mean(connectivities):.1f}% ± {np.std(connectivities):.1f}%\n")


if __name__ == "__main__":
    validate_genome("best_genome_new2.npy", "config.json", "baseline")
    #validate_genome("best_genome_no_lvd.npy", "config.json", "no_lvd")
    #validate_genome("best_genome_no_rel_pos.npy", "config.json", "no_rel_pos")
    #validate_genome("best_genome_no_time.npy", "config.json", "no_time")
    #validate_genome("best_genome_no_target_dir.npy", "config.json", "no_target_dir")
