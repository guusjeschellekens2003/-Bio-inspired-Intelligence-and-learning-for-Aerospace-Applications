import numpy as np


#choice your experiment
# Available options: "baseline", "no_lvd", "no_time", "no_rel_pos", "no_target_dir"
CURRENT_EXPERIMENT = "baseline"

# Terrain Generation — fully vectorised

def generate_terrain(config, seed=None):

    if seed is not None:
        rng = np.random.default_rng(int(seed))
    else:
        rng = np.random.default_rng()

    gs    = config["grid_size"]
    scale = config.get("terrain_scale", 8.0)

    i_idx = np.arange(gs)[:, np.newaxis]
    j_idx = np.arange(gs)[np.newaxis, :]
    x = i_idx / scale
    y = j_idx / scale

    terrain   = np.zeros((gs, gs))
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    for _ in range(4):
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        terrain   += amplitude * np.sin(frequency * x + px) * np.cos(frequency * y + py)
        max_value += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    terrain /= max_value

    target    = config.get("obstacle_fraction", 0.20)
    thresh    = np.percentile(terrain, (1.0 - target) * 100)
    obstacles = terrain > thresh

    # Keep spawn area (centre) clear
    world = config["world_size"]
    cell  = config["cell_size"]
    cx = int(np.clip(world / cell, 0, gs - 1))
    cy = int(np.clip(world / cell, 0, gs - 1))
    r  = 3
    obstacles[max(0, cx-r):cx+r+1, max(0, cy-r):cy+r+1] = False
    return obstacles


_LVD_R = 3
_lvd_offsets = np.arange(2 * _LVD_R + 1) - _LVD_R
_LVD_DI, _LVD_DJ = np.meshgrid(_lvd_offsets, _lvd_offsets, indexing='ij')
_LVD_DI = _LVD_DI.ravel()
_LVD_DJ = _LVD_DJ.ravel()


def least_visited_direction(positions, grids, config):
    """
    from Engebraaten et al. (2019).
    Unit vector toward nearest unvisited cell in 7x7 local window.
    """
    world     = config["world_size"]
    cell      = config["cell_size"]
    gs        = config["grid_size"]
    r         = _LVD_R
    n_genomes, n_agents, _ = positions.shape

    ax = np.clip(((positions[:,:,0]+world)/cell).astype(int), 0, gs-1)
    ay = np.clip(((positions[:,:,1]+world)/cell).astype(int), 0, gs-1)

    inv    = 1.0 - grids
    padded = np.pad(inv, ((0,0),(r,r),(r,r)),
                    mode='constant', constant_values=0.0)

    xi = (ax+r)[:,:,np.newaxis] + _LVD_DI[np.newaxis,np.newaxis,:]
    yi = (ay+r)[:,:,np.newaxis] + _LVD_DJ[np.newaxis,np.newaxis,:]
    g_idx = np.arange(n_genomes)[:,np.newaxis,np.newaxis]

    best_p  = np.argmax(padded[g_idx, xi, yi], axis=2)
    best_di = _LVD_DI[best_p]
    best_dj = _LVD_DJ[best_p]

    tx = (ax + best_di + 0.5)*cell - world - positions[:,:,0]
    ty = (ay + best_dj + 0.5)*cell - world - positions[:,:,1]
    norm = np.sqrt(tx**2 + ty**2)
    norm = np.where(norm == 0, 1.0, norm)
    return np.stack([tx/norm, ty/norm], axis=2)



# Obstacle repulsion — magnitude-weighted

def obstacle_repulsion(positions, obstacles, config):

    world = config["world_size"]
    cell  = config["cell_size"]
    gs    = config["grid_size"]
    r     = _LVD_R

    n_genomes, n_agents, _ = positions.shape
    ax = np.clip(((positions[:,:,0]+world)/cell).astype(int), 0, gs-1)
    ay = np.clip(((positions[:,:,1]+world)/cell).astype(int), 0, gs-1)

    padded = np.pad(obstacles.astype(float), ((r,r),(r,r)),
                    mode='constant', constant_values=0.0)

    offsets = np.arange(2*r+1) - r
    di, dj  = np.meshgrid(offsets, offsets, indexing='ij')
    di = di.ravel(); dj = dj.ravel()

    xi = (ax[:,:,np.newaxis] + r) + di[np.newaxis,np.newaxis,:]
    yi = (ay[:,:,np.newaxis] + r) + dj[np.newaxis,np.newaxis,:]
    patch = padded[xi, yi]

    dist   = np.sqrt(di**2 + dj**2 + 1e-8)
    weight = patch / (dist[np.newaxis,np.newaxis,:] + 1e-8)
    rx = np.sum(weight * (di/dist)[np.newaxis,np.newaxis,:], axis=2)
    ry = np.sum(weight * (dj/dist)[np.newaxis,np.newaxis,:], axis=2)

    # Keep magnitude (encodes proximity) but clip to [0,1]
    magnitude   = np.sqrt(rx**2 + ry**2)
    mag_clipped = np.clip(magnitude / 5.0, 0.0, 1.0)
    safe_mag    = np.where(magnitude == 0, 1.0, magnitude)
    direction   = np.stack([rx/safe_mag, ry/safe_mag], axis=2)
    return direction * mag_clipped[:,:,np.newaxis]   # (G, A, 2)


# Neural network controller

def population_action(config, population, inputs):
    """Single hidden-layer tanh network."""
    n_hidden  = config["n_hidden"]
    n_inputs  = config["n_inputs"]
    n_outputs = config["n_outputs"]

    w1 = population[:, :n_hidden*n_inputs].reshape(
         population.shape[0], n_inputs, n_hidden)
    b1 = population[:, n_hidden*n_inputs:n_hidden*(n_inputs+1)].reshape(
         population.shape[0], 1, n_hidden)
    w2 = population[:, n_hidden*(n_inputs+1):n_hidden*(n_inputs+1+n_outputs)].reshape(
         population.shape[0], n_hidden, n_outputs)
    b2 = population[:, -n_outputs:].reshape(
         population.shape[0], 1, n_outputs)

    z1 = np.einsum("ijk,ikl->ijl", inputs, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.einsum("ijk,ikl->ijl", a1, w2) + b2
    return np.tanh(z2)



# Grid update

def update_grids(grids, positions, obstacles, config):
    """Mark visited FREE cells. Obstacle cells never counted as explored."""
    world = config["world_size"]
    cell  = config["cell_size"]
    gs    = config["grid_size"]
    n_genomes, n_agents, _ = positions.shape

    xs = np.clip(((positions[:,:,0]+world)/cell).astype(int), 0, gs-1)
    ys = np.clip(((positions[:,:,1]+world)/cell).astype(int), 0, gs-1)

    g_idx   = np.repeat(np.arange(n_genomes), n_agents)
    x_flat  = xs.ravel(); y_flat = ys.ravel()
    is_free = ~obstacles[x_flat, y_flat]
    grids[g_idx[is_free], x_flat[is_free], y_flat[is_free]] = 1
    return xs, ys



# Input normalisation

def normalise_inputs(inputs, config):
    """Normalise vel and rel_pos channels to [-1, 1]."""
    n_agents = config["n_agents"]
    world    = config["world_size"]
    v_max    = config["v_max_follower"]

    normed = inputs.copy()
    normed[:,:, 0:2] /= (v_max + 1e-8)
    rel_end = 2 + (n_agents-1)*2
    normed[:,:, 2:rel_end] /= (2*world + 1e-8)
    return normed



# MAIN SIMULATION

def single_evaluate(config, population, return_components=False,
                    analysis=False, terrain_seed=None):

    if terrain_seed is not None:
        np.random.seed(terrain_seed)

    n_agents    = config["n_agents"]
    n_followers = n_agents - 1
    n_genomes   = population.shape[0]
    loops       = int(config["t_max"] / config["dt"])
    world       = config["world_size"]

    obstacles = generate_terrain(config, seed=terrain_seed)


    leader_pos = np.zeros((n_genomes, 1, 2))

    ring_rotation = np.random.uniform(0, 2*np.pi)
    spread_r      = 0.3
    angles_init   = (np.linspace(0, 2*np.pi, n_followers, endpoint=False)
                     + ring_rotation)
    follower_pos  = np.stack([spread_r * np.cos(angles_init),
                              spread_r * np.sin(angles_init)], axis=1)
    follower_pos  = np.tile(follower_pos[np.newaxis], (n_genomes, 1, 1))
    follower_pos += np.random.randn(n_genomes, n_followers, 2) * 0.05
    positions     = np.concatenate([leader_pos, follower_pos], axis=1)

    leader_vel   = np.zeros((n_genomes, 1, 2))
    follower_vel = np.stack([np.cos(angles_init),
                             np.sin(angles_init)], axis=1)
    follower_vel = np.tile(follower_vel[np.newaxis], (n_genomes, 1, 1))
    follower_vel = follower_vel * (1.0 * config["v_max_follower"])
    velocity     = np.concatenate([leader_vel, follower_vel], axis=1)

    grids = np.zeros((n_genomes, config["grid_size"], config["grid_size"]))

    v_max_arr = np.zeros((1, n_agents, 1))
    v_max_arr[0, 0, 0]  = config["v_max_leader"]
    v_max_arr[0, 1:, 0] = config["v_max_follower"]

    a_max_arr = np.zeros((1, n_agents, 1))
    a_max_arr[0, 0, 0]  = config["a_max_leader"]
    a_max_arr[0, 1:, 0] = config["a_max_follower"]


    sector_dirs = {}
    for j in range(1, n_agents):
        angle = (j - 1) * (2 * np.pi / n_followers)
        sector_dirs[j] = np.array([np.cos(angle), np.sin(angle)])

    # Connectivity: trailing window
    WINDOW_FRAC  = config.get("connectivity_window_fraction", 0.125)
    WINDOW_STEPS = max(1, int(WINDOW_FRAC * loops))
    window_start = loops - WINDOW_STEPS
    window_connected = np.zeros((n_genomes, n_followers), dtype=bool)


    total_obstacle_hits = np.zeros((n_genomes, n_agents), dtype=float)
    total_col_hits      = np.zeros(n_genomes, dtype=float)
    max_pairs           = n_agents * (n_agents - 1) / 2
    total_distance      = np.zeros((n_genomes, n_agents), dtype=float)

    if analysis:
        position_history = np.zeros((loops, n_genomes, n_agents, 2))

    LVD_INTERVAL = 10
    lvd     = np.zeros((n_genomes, n_agents, 2))
    obs_rep = np.zeros((n_genomes, n_agents, 2))

    for step in range(loops):

        if analysis:
            position_history[step] = positions.copy()

        xs, ys = update_grids(grids, positions, obstacles, config)
        total_obstacle_hits += obstacles[xs, ys].astype(float)

        old_positions = positions.copy()

        # Connectivity window check
        if step >= window_start:
            comm_range = config["comm_range_leader"]
            leader_p   = positions[:, 0:1, :]
            follower_p = positions[:, 1:,  :]
            dists_now  = np.linalg.norm(follower_p - leader_p, axis=2)
            window_connected |= (dists_now < comm_range)


        if step % LVD_INTERVAL == 0:
            lvd     = least_visited_direction(positions, grids, config)
            obs_rep = obstacle_repulsion(positions, obstacles, config)

        time_fraction = step / loops

        # Build inputs
        positions_centered = positions - np.mean(positions, axis=1, keepdims=True)
        agent_inputs = []
        for j in range(n_agents):
            others  = np.delete(np.arange(n_agents), j)
            rel_pos = positions_centered[:, others, :].reshape(n_genomes, 1, -1)
            own_vel = velocity[:, j, :].reshape(n_genomes, 1, 2)
            flag    = np.full((n_genomes, 1, 1), 1.0 if j == 0 else 0.0)
            lvd_j   = lvd[:, j, :].reshape(n_genomes, 1, 2)
            obs_j   = obs_rep[:, j, :].reshape(n_genomes, 1, 2)
            t_inp   = np.full((n_genomes, 1, 1), time_fraction)


            wall_j  = (positions[:, j, :] / world).reshape(n_genomes, 1, 2)

            if j == 0:
                leader_rel = np.zeros((n_genomes, 1, 2))
                identity   = np.full((n_genomes, 1, 1), 0.0)
                target_dir = np.zeros((n_genomes, 1, 2))
            else:
                lr = positions[:, 0, :] - positions[:, j, :]
                lr_norm    = np.clip(lr / (config["comm_range_leader"] + 1e-8),
                                     -1.0, 1.0)
                leader_rel = lr_norm.reshape(n_genomes, 1, 2)
                identity   = np.full((n_genomes, 1, 1), j / (n_agents - 1))
                target_dir = np.full((n_genomes, 1, 2), sector_dirs[j])

            if CURRENT_EXPERIMENT == "baseline":
                agent_inputs.append(
                    np.concatenate([own_vel, rel_pos, flag,
                                    lvd_j, obs_j, target_dir, leader_rel,
                                    wall_j, t_inp], axis=2)
                )

            elif CURRENT_EXPERIMENT == "no_lvd":
                agent_inputs.append(
                    np.concatenate([own_vel, rel_pos, flag,
                                    obs_j, target_dir, leader_rel,
                                    wall_j, t_inp], axis=2)
                )

            #check
            elif CURRENT_EXPERIMENT == "no_target_dir":
                agent_inputs.append(
                    np.concatenate([own_vel, flag, rel_pos,
                                    obs_j, leader_rel, lvd_j,
                                    wall_j, t_inp], axis=2)
                )

            elif CURRENT_EXPERIMENT == "no_time":
                agent_inputs.append(
                    np.concatenate([own_vel, flag, rel_pos,
                                    obs_j, target_dir, leader_rel, lvd_j,
                                    wall_j], axis=2)
                )

            elif CURRENT_EXPERIMENT == "no_rel_pos":
                agent_inputs.append(
                    np.concatenate([own_vel, flag,
                                    obs_j, target_dir, leader_rel, lvd_j,
                                    wall_j, t_inp], axis=2)
                )

        inputs = np.concatenate(agent_inputs, axis=1)
        inputs = normalise_inputs(inputs, config)

        accel     = population_action(config, population, inputs) * a_max_arr
        velocity += accel * config["dt"]

        speed      = np.linalg.norm(velocity, axis=2, keepdims=True)
        safe_speed = np.where(speed == 0, 1.0, speed)
        velocity   = np.where(speed < v_max_arr,
                              velocity, velocity / safe_speed * v_max_arr)


        velocity[:, 0, :] *= 0.85


        total_distance += np.linalg.norm(velocity, axis=2) * config["dt"]

        positions += velocity * config["dt"]

        # Reflective world boundaries
        out_of_bounds = np.abs(positions) > world
        positions     = np.clip(positions, -world, world)
        velocity      = np.where(out_of_bounds, -velocity * 0.5, velocity)

        # Rock collision — axis-aligned reflection
        cx_new = np.clip(((positions[:,:,0]+world)/config["cell_size"]).astype(int),
                         0, config["grid_size"]-1)
        cy_new = np.clip(((positions[:,:,1]+world)/config["cell_size"]).astype(int),
                         0, config["grid_size"]-1)
        hits = obstacles[cx_new, cy_new]

        if np.any(hits):
            hits_exp  = hits[:,:,np.newaxis]
            positions = np.where(hits_exp, old_positions, positions)

            tx = old_positions.copy()
            tx[:,:,0] += velocity[:,:,0] * config["dt"]
            cxt = np.clip(((tx[:,:,0]+world)/config["cell_size"]).astype(int),
                          0, config["grid_size"]-1)
            cyt = np.clip(((tx[:,:,1]+world)/config["cell_size"]).astype(int),
                          0, config["grid_size"]-1)
            hit_x = obstacles[cxt, cyt][:,:,np.newaxis]

            ty = old_positions.copy()
            ty[:,:,1] += velocity[:,:,1] * config["dt"]
            cxt2 = np.clip(((ty[:,:,0]+world)/config["cell_size"]).astype(int),
                           0, config["grid_size"]-1)
            cyt2 = np.clip(((ty[:,:,1]+world)/config["cell_size"]).astype(int),
                           0, config["grid_size"]-1)
            hit_y = obstacles[cxt2, cyt2][:,:,np.newaxis]

            vx = velocity[:,:,0:1]
            vy = velocity[:,:,1:2]
            vx = np.where(hits_exp & hit_x, -vx * 0.7, vx)
            vy = np.where(hits_exp & hit_y, -vy * 0.7, vy)
            velocity = np.concatenate([vx, vy], axis=2)

            # Safety: corner-stuck agents
            cx_f = np.clip(((positions[:,:,0]+world)/config["cell_size"]).astype(int),
                           0, config["grid_size"]-1)
            cy_f = np.clip(((positions[:,:,1]+world)/config["cell_size"]).astype(int),
                           0, config["grid_size"]-1)
            stuck     = obstacles[cx_f, cy_f][:,:,np.newaxis]
            positions = np.where(stuck, old_positions, positions)
            velocity  = np.where(stuck, -velocity * 0.3, velocity)

        # Soft agent-agent repulsion
        d_min = config["d_min"]
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                diff      = positions[:,i,:] - positions[:,j,:]
                dist      = np.linalg.norm(diff, axis=1, keepdims=True)
                too_close = (dist < d_min) & (dist > 0)
                safe_d    = np.where(dist == 0, 1.0, dist)
                push      = np.where(too_close,
                                     (d_min-dist)/2*diff/safe_d, 0.0)
                positions[:,i,:] += push
                positions[:,j,:] -= push

        # Collision penalty accumulated every step
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                dists = np.linalg.norm(
                    positions[:,i,:] - positions[:,j,:], axis=1)
                total_col_hits += (dists < d_min).astype(float)



    # Coverage: fraction of free cells visited
    free_cells = max(1, int(grids[0].size - np.sum(obstacles)))
    coverage   = np.sum(grids.reshape(n_genomes,-1), axis=1) / free_cells

    # Connectivity: fraction of scouts that uplinked during trailing window
    connect = np.mean(window_connected.astype(float), axis=1)


    max_dist = np.zeros((1, n_agents))
    max_dist[0, 0]  = config["v_max_leader"]   * config["t_max"]
    max_dist[0, 1:] = config["v_max_follower"] * config["t_max"]
    energy_norm = np.mean(total_distance / (max_dist + 1e-8), axis=1)

    # Obstacle hits: fraction of agent-timesteps inside rocks [0,1]
    obs_hits = np.mean(total_obstacle_hits / loops, axis=1)

    # Collision penalty: fraction of pair-timesteps with overlap [0,1]
    col_pen = (total_col_hits / max_pairs) / loops

    # Fitness — lower is better (CMA-ES minimises)
    fitness = (
        3.0 * (1.0 - coverage)  +
        1.5 * (1.0 - connect)   +
        0.3 * energy_norm        +
        3.0 * col_pen            +
        3.0 * obs_hits
    )

    if analysis:
        return fitness, coverage, connect, position_history, obstacles
    if return_components:
        return fitness, coverage, connect, energy_norm, col_pen, obs_hits
    return fitness


def evaluate_population(config, population):
    """Average fitness over n_evals random episodes to reduce noise."""
    fitnesses = np.zeros(population.shape[0])
    for _ in range(config["n_evals"]):
        fitnesses += single_evaluate(config, population)
    return fitnesses / config["n_evals"]