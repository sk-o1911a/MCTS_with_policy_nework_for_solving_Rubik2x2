import numpy as np
import torch

from Rubik2x2Env import Rubik2x2Env
from MCTS_Core import MCTS
from Action_MCTS import pick_action_from_mcts


def generate_self_play_data(
    model,
    num_episodes: int = 10,
    max_episode_steps: int = 25,
    num_simulations: int = 200,
    scramble_len: int = 4,
    select_mode: str = "sample",
    temperature: float = 1.0,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = Rubik2x2Env(scramble_len=scramble_len)

    mcts = MCTS(
        model=model,
        num_actions=env.action_space.n,
        num_simulations=num_simulations,
        device=device,
    )

    dataset: list[tuple[np.ndarray, np.ndarray, float]] = []

    solved_count = 0
    total_episodes = num_episodes

    for ep in range(num_episodes):
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[self-play] episode {ep + 1}/{num_episodes} ...")
        obs, info = env.reset()
        cube = env.cube
        action_mask = info.get("action_mask", None)

        episode_states: list[np.ndarray] = []
        episode_pis: list[np.ndarray] = []
        solved = False

        for step in range(max_episode_steps):
            visit_counts = mcts.run(cube, obs, action_mask)

            vc_sum = visit_counts.sum()
            if vc_sum == 0:
                pi = np.ones_like(visit_counts, dtype=np.float32) / len(visit_counts)
            else:
                pi = visit_counts / vc_sum

            episode_states.append(obs.copy())
            episode_pis.append(pi.copy())

            action = pick_action_from_mcts(
                visit_counts,
                mode=select_mode,
                temperature=temperature,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            cube = env.cube
            action_mask = info.get("action_mask", None)

            if terminated:
                solved = True
                break

        if solved:
            solved_count += 1
            steps_used = step + 1

            z = 1.0 - 0.05 * steps_used
            if z < 0.2:
                z = 0.2
        else:
            z = -1.0

        for s, pi in zip(episode_states, episode_pis):
            dataset.append((s, pi, z))

    solve_rate = solved_count / total_episodes
    print(f"[self-play] solve rate: {solve_rate*100.0:.1f}% ({solved_count}/{total_episodes})")
    return dataset, solve_rate
