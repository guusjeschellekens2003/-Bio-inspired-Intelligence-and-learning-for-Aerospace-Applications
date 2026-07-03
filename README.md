# -Bio-inspired-Intelligence-and-learning-for-Aerospace-Applications
Multi-Rover Intelligence for Mars Exploration

# Evolutionary Heterogeneous Mars Swarm Exploration

An evolutionary robotics framework that uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to evolve decentralised neural controllers for a heterogeneous Mars rover system. The system consists of one central leader (acting as a science base/relay) and multiple smaller scout rovers optimizing for maximum terrain coverage and network connectivity.

## Repository Structure

### Core Evolutionary Engine
* **`train2.py`**: The primary training script using CMA-ES to evolve neural network weights across generations.
* **`evaluation.py`**: Evaluates evolved controllers against the 2D particle simulation map, tracking area coverage, collision avoidance, and uplink integrity.
* **`validation.py`**: Benchmarks controller robustness against the standardised terrain layouts.
* **`pareto_sweep.py`**: Shifting fitness weights between $w_{cov}$ and $w_{con}$ to map out the operational limits.

### Ablation Studies & Genomes (`.npy`)
Pre-trained controller checkpoints mapping the neural architecture input configuration experiments:
* **`best_genome_baseline.npy`**: Full operational input sensor suite.
* **`best_genome_no_lvd.npy`**: Least Visited Direction vector sliced out.
* **`best_genome_no_rel_pos.npy`**: Tracking of other scout positions disabled.
* **`best_genome_no_target_dir.npy`**: Target orientation vector removed.
* **`best_genome_no_time.npy`**: Mission clock fraction removed.

###  Analysis & Sensitivity
* **`Sensitivity.py` / `Sensitivity2.py`**: Code evaluating system performance against scaling variables (e.g., expanding swarm size, shifting communication ranges).
* **`visualisation2.py`**: Post-evaluation rendering script that plots learning curves, maps agent trajectories, and outputs pathing maps.

###  Results & Outputs
* **`exploration_result2.png` / `result3.png`**: Static pathing and coverage map readouts.
* **`swarm_exploration.gif` (1, 2, 3)**: Rendered animations tracking real-time swarm trajectory developments and leader communication envelopes.

---

