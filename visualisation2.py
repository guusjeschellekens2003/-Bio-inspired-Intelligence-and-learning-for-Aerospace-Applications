"""
visualise.py — Analyse, plot, and animate the best evolved genome.
Run AFTER train.py has saved best_genome.npy.
"""
import numpy as np
import json5
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import opensimplex
from evaluation import single_evaluate

#choice experiment
CURRENT_EXPERIMENT = "baseline"

def load_config():
    """Must be identical to train.py's config block."""
    with open("config.json") as f:
        config = json5.load(f)
    config["cell_size"]      = (2 * config["world_size"]) / config["grid_size"]

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

    config["n_outputs"]      = 2
    config["n_param"] = (
        config["n_hidden"] * config["n_inputs"]  +
        config["n_hidden"]                       +
        config["n_hidden"] * config["n_outputs"] +
        config["n_outputs"]
    )
    return config



def learning_curve():
    """Loads training history and plots f_min and f_avg over generations."""
    try:
        history = np.load("learning_history.npy")
    except FileNotFoundError:
        print("Warning: 'learning_history.npy' not found.")
        return

    gens = history[:, 0]
    f_min = history[:, 1]
    f_mean = history[:, 2]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Best Fitness (Teal) and Average Fitness (Orange)
    ax.plot(gens, f_min, label='Best Fitness ($f_{min}$)', color='teal', linewidth=2)
    ax.plot(gens, f_mean, label='Average Fitness ($f_{avg}$)', color='darkorange', alpha=0.8, linestyle='--')

    ax.set_title("Optimization Learning Curve: Baseline", fontsize=24, pad=20)
    ax.set_xlabel("Generation", fontsize=20, labelpad=15)
    ax.set_ylabel("Fitness", fontsize=20, labelpad=15)

    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig("learning_curve_baseline.png", dpi=150)
    print("Saved learning_curve_baseline.png")
    plt.show()


def apply_scale_ticks(ax, config, axis='both'):

    scale = config.get("scale_metres_per_unit", 25.0)
    world = config["world_size"]
    ticks = np.linspace(-world, world, 5)
    tick_labels = [f"{t * scale:.0f}m" for t in ticks]

    if axis in ('both', 'x'):
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
    if axis in ('both', 'y'):
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)


def create_animation(config, ph, obstacles, colors, labels):
    """Generates and saves a high-speed GIF tracking the swarm over time."""
    scale = config.get("scale_metres_per_unit", 25.0)
    world = config["world_size"]
    n_agents = config["n_agents"]
    total_frames = ph.shape[0]


    frame_skip = 5
    frames_to_render = list(range(0, total_frames, frame_skip))
    if frames_to_render[-1] != total_frames - 1:
        frames_to_render.append(total_frames - 1)  # Make sure we capture the final frame

    print(f"Generating swarm exploration video (Rendering {len(frames_to_render)}/{total_frames} frames)...")

    fig, ax = plt.subplots(figsize=(6, 6))  # Slightly smaller canvas = faster pixel processing
    comm_range = config["comm_range_leader"]

    def update(frame):
        ax.clear()


        ax.imshow(obstacles.T, origin="lower", cmap="Greys", alpha=0.4,
                  extent=[-world, world, -world, world])


        leader_pos = ph[frame, 0]
        comm_circle = plt.Circle(
            (leader_pos[0], leader_pos[1]), comm_range,
            color="red", fill=False, linestyle="--", linewidth=1.5, alpha=0.6
        )
        ax.add_patch(comm_circle)


        for i in range(n_agents):
            if frame > 0:
                ax.plot(ph[:frame + 1, i, 0], ph[:frame + 1, i, 1],
                        color=colors[i], lw=1.2, alpha=0.7)

            marker = 's' if i == 0 else 'o'
            size = 80 if i == 0 else 45
            ax.scatter(ph[frame, i, 0], ph[frame, i, 1],
                       color=colors[i], marker=marker, s=size, edgecolors='black', zorder=5)


        ax.set_xlim(-world, world)
        ax.set_ylim(-world, world)
        ax.set_aspect("equal")
        apply_scale_ticks(ax, config)

        ax.set_title(f"Mars Swarm Simulation — Step {frame}/{total_frames}\n"
                     f"Time Elapsed: {frame * 0.1:.1f}s", fontsize=10)
        ax.set_xlabel(f"x ({scale:.0f} m/unit)")
        ax.set_ylabel(f"y ({scale:.0f} m/unit)")

        legend_elements = [
            mpatches.Patch(color='red', label='Leader'),
            mpatches.Patch(color='C1', label='Scouts'),
            mpatches.Patch(color='none', label='Dashed Circle = Comm Range')
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)


    ani = animation.FuncAnimation(fig, update, frames=frames_to_render, interval=60)

    video_filename = "swarm_exploration.mp4"

    ani.save("swarm_exploration.gif", writer="pillow", fps=15, dpi=75)
    print(" Success! Optimized GIF saved as 'swarm_exploration.gif'")

    plt.close(fig)


def main():
    config = load_config()
    scale  = config.get("scale_metres_per_unit", 25.0)
    world  = config["world_size"]


    opensimplex.seed(12345)
    terrain_seed = 50
    #learning_curve()
    try:
        genome = np.load(f"best_genome_new2.npy").reshape(1, -1)
    except FileNotFoundError:
        print("Error: 'best_genome.npy' not found. Please run your training script first!")
        return

    # Evaluate the genome
    fitness, coverage, connect, pos_hist, obstacles = single_evaluate(
        config, genome, analysis=True, terrain_seed=terrain_seed
    )

    print(f"Fitness      : {fitness[0]:.4f}")
    print(f"Coverage     : {coverage[0]*100:.1f}%  of free cells")
    print(f"Connectivity : {connect[0]*100:.1f}%  of scouts in leader range")
    print(f"World scale  : 1 unit = {scale}m  →  "
          f"world = {world*2*scale:.0f}m × {world*2*scale:.0f}m")

    # Treat history structures
    ph       = pos_hist[:, 0, :, :]   # (loops, n_agents, 2)
    n_agents = config["n_agents"]
    colors   = ["red"] + [f"C{i}" for i in range(1, n_agents)]
    labels   = ["Leader (Perseverance-style)"] + [f"Scout {i}" for i in range(1, n_agents)]

    create_animation(config, ph, obstacles, colors, labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Mars Swarm Exploration — Best Evolved Controller\n"
        f"World: {world*2*scale:.0f}m × {world*2*scale:.0f}m  |  "
        f"Coverage: {coverage[0]*100:.1f}%  |  "
        f"Connectivity: {connect[0]*100:.1f}%",
        fontsize=12
    )

    ax = axes[0]
    ax.set_xlim(-world, world)
    ax.set_ylim(-world, world)
    ax.set_aspect("equal")
    ax.set_title("Agent Trajectories", fontsize=20, pad=15)

    ax.imshow(obstacles.T, origin="lower", cmap="Greys", alpha=0.3,
              extent=[-world, world, -world, world])

    for i in range(n_agents):
        color = 'red' if i == 0 else 'teal'
        ax.plot(ph[:, i, 0], ph[:, i, 1], color=color, lw=1.2, alpha=0.7)
        ax.scatter(ph[0, i, 0], ph[0, i, 1], color=color, marker="o", s=80, zorder=5, edgecolor='black')
        ax.scatter(ph[-1, i, 0], ph[-1, i, 1], color=color, marker="*", s=150, zorder=5, edgecolor='black')


    ax.add_patch(plt.Circle((ph[0, 0, 0], ph[0, 0, 1]), config["comm_range_leader"],
                            color="red", fill=False, linestyle=":", linewidth=2, alpha=0.4))
    ax.add_patch(plt.Circle((ph[-1, 0, 0], ph[-1, 0, 1]), config["comm_range_leader"],
                            color="red", fill=False, linestyle="--", linewidth=2, alpha=0.8))


    ax2 = axes[1]
    gs = config["grid_size"]
    cell = config["cell_size"]

    grid = np.zeros((gs, gs))
    for t in range(ph.shape[0]):
        for a in range(n_agents):
            xi = int(np.clip((ph[t, a, 0] + world) / cell, 0, gs - 1))
            yi = int(np.clip((ph[t, a, 1] + world) / cell, 0, gs - 1))
            grid[xi, yi] = 1

    # Define colours: Dark Grey (Obstacles), White (Unexplored), Dark Orange (Visited)
    display = np.ones((gs, gs, 3))
    display[obstacles] = [0.4, 0.4, 0.4]
    display[grid.astype(bool) & ~obstacles] = [255 / 255, 140 / 255, 0 / 255]  # Dark Orange

    ax2.imshow(display.transpose(1, 0, 2), origin="lower", extent=[-world, world, -world, world])
    ax2.set_aspect("equal")
    ax2.set_title(f"Coverage: {coverage[0] * 100:.1f}%", fontsize=20, pad=15)

    # Apply global styling
    for a in [ax, ax2]:
        apply_scale_ticks(a, config, axis='both')
        a.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.savefig("exploration_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()