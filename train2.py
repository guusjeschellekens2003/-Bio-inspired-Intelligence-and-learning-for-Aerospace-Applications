import numpy as np
import json5

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import Problem
from evaluation2 import evaluate_population


# Available options: "baseline", "no_lvd", "no_time", "no_rel_pos", "no_target_dir"

CURRENT_EXPERIMENT = "no_target_dir"


class SwarmProblem(Problem):

    def __init__(self, config):
        self.config = config
        self.gen = 0
        self.history = []
        super().__init__(
            n_var=config["n_param"],
            n_obj=1,
            n_constr=0,
            xl=-100 * np.ones(config["n_param"]),
            xu=+100 * np.ones(config["n_param"]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f = evaluate_population(self.config, x)
        out["F"] = np.asarray(f).reshape(-1)
        self.history.append((self.gen, float(np.min(f)), float(np.mean(f))))
        self.gen += 1


def train():
    with open("config.json") as f:
        config = json5.load(f)

    # Attach the chosen experiment variant to the configuration profile
    config["ablation_type"] = CURRENT_EXPERIMENT
    config["cell_size"] = (2 * config["world_size"]) / config["grid_size"]


    if CURRENT_EXPERIMENT == "baseline":
        config["n_inputs"] = 2 * config["n_agents"] + 12
        breakdown_str = f"vel(2) + rel_pos({(config['n_agents'] - 1) * 2}) + flag(1) + lvd(2) + obs_rep(2) + target(2) + leader(2) + wall(2) + time(1)"

    elif CURRENT_EXPERIMENT == "no_lvd" or CURRENT_EXPERIMENT == "no_target_dir" :
        config["n_inputs"] = 2 * config["n_agents"] + 10  # Slices out lvd (2)
        breakdown_str = f"vel(2) + rel_pos({(config['n_agents'] - 1) * 2}) + flag(1) + obs_rep(2) + target(2) + leader(2) + wall(2) + time(1)"

    elif CURRENT_EXPERIMENT == "no_time":
        config["n_inputs"] = 2 * config["n_agents"] + 11  # Slices out time_fraction (1)
        breakdown_str = f"vel(2) + rel_pos({(config['n_agents'] - 1) * 2}) + flag(1) + lvd(2) + obs_rep(2) + target(2) + leader(2) + wall(2)"

    elif CURRENT_EXPERIMENT == "no_rel_pos":
        # Slices out the entire relative position vector tracking other scouts ((N-1)*2)
        # Constant tracking size: vel(2)+flag(1)+lvd(2)+obs(2)+target(2)+leader(2)+wall(2)+time(1) = 14
        config["n_inputs"] = 14
        breakdown_str = "vel(2) + flag(1) + lvd(2) + obs_rep(2) + target(2) + leader(2) + wall(2) + time(1)"

    else:
        raise ValueError(f"Invalid assignment choice: '{CURRENT_EXPERIMENT}'. Check your global switch string setting.")

    config["n_outputs"] = 2


    config["n_param"] = (
            config["n_hidden"] * config["n_inputs"] +  # W1
            config["n_hidden"] +  # b1
            config["n_hidden"] * config["n_outputs"] +  # W2
            config["n_outputs"]  # b2
    )

    print("=" * 60)
    print(f"ACTIVE EXPERIMENT CONFIG : {CURRENT_EXPERIMENT.upper()}")
    print(f"Network parameters       : {config['n_param']}")
    print(f"Input nodes (n_inputs)   : {config['n_inputs']}")
    print(f"Input Breakdown          : {breakdown_str}")
    print(f"Total Agents managed     : {config['n_agents']}")
    print(f"Generation limits        : {config.get('n_gen', 2000)}")
    print("=" * 60)

    problem = SwarmProblem(config)
    algorithm = CMAES(
        x0=np.random.randn(config["n_param"]) * 0.1,
        sigma=0.2,
        popsize=config["n_genomes"],
        restarts=0,
    )

    termination = DefaultSingleObjectiveTermination(
        ftol=1e-6,
        n_max_gen=config.get("n_gen", 2000),
    )

    # Run the optimization loop
    res = minimize(problem, algorithm, termination=termination, verbose=True)


    print(f"\nBest fitness achieved : {res.F[0]:.4f}")

    output_genome_name = f"best_genome_{CURRENT_EXPERIMENT}.npy"
    output_history_name = f"learning_history_{CURRENT_EXPERIMENT}.npy"

    np.save(output_genome_name, res.X)
    np.save(output_history_name, np.array(problem.history))

    print(f"Successfully saved tracking weight matrix as: '{output_genome_name}'")
    print(f"Successfully saved tracking convergence curves as: '{output_history_name}'")


if __name__ == "__main__":
    train()