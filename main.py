import os
import torch
from sympy.physics.units import temperature

from Policy_Value_Net import PolicyValueNet
from Self_Play import generate_self_play_data
from Train_Network import train_on_selfplay_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


NUM_ITERS = 300
EPISODES_PER_ITER = 70
EPOCHS = 7
BATCH_SIZE = 256
LR = 1e-4
SOLVE_THRESHOLD = 0.9
CHECKPOINT_PATH = "rubik_policy_value.pt"


def load_or_create_model(device=DEVICE):
    model = PolicyValueNet().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[main] found checkpoint: {CHECKPOINT_PATH}, loading...")
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)
    else:
        print("[main] no checkpoint found, create new model.")
    model.eval()
    return model

def get_temperature(scramble_len: int) -> float:
    if scramble_len <= 4:
        return 1.0
    elif scramble_len <= 7:
        return 0.7
    elif scramble_len <= 9:
        return 0.4
    else:
        return 0.3

def main():
    device = DEVICE
    model = load_or_create_model(device)

    SCRAMBLE_LEN = 1
    recent_solve_rates: list[float] = []

    for it in range(1, NUM_ITERS + 1):
        if SCRAMBLE_LEN <= 3:
            SIMULATIONS = 300
        elif SCRAMBLE_LEN <= 6:
            SIMULATIONS = 400
        else:
            SIMULATIONS = 500

        temperature = get_temperature(SCRAMBLE_LEN)

        print(f"\n=== Iteration {it}/{NUM_ITERS} (scramble_len={SCRAMBLE_LEN}, sims={SIMULATIONS}), current temp={temperature} ===")

        dataset, solve_rate = generate_self_play_data(
            model=model,
            num_episodes=EPISODES_PER_ITER,
            max_episode_steps=30,
            num_simulations=SIMULATIONS,
            scramble_len=SCRAMBLE_LEN,
            select_mode="sample",
            temperature=temperature,
            device=device,
        )
        print(f"[main] self-play collected {len(dataset)} samples.")

        recent_solve_rates.append(solve_rate)
        if len(recent_solve_rates) > 5:
            recent_solve_rates.pop(0)

        model = train_on_selfplay_data(
            model,
            dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            device=device,
        )

        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"[main] saved checkpoint to {CHECKPOINT_PATH}")
        model.eval()

        if len(recent_solve_rates) == 5:
            avg_solve = sum(recent_solve_rates) / 5.0
            print(f"[main] avg_solve(last 5) = {avg_solve * 100:.1f}%")
            if avg_solve >= SOLVE_THRESHOLD and SCRAMBLE_LEN < 10:
                SCRAMBLE_LEN += 1
                recent_solve_rates.clear()
                print(f"[main] unlock next scramble_len = {SCRAMBLE_LEN}")
            else:
                print("[main] stay at current scramble_len")

    print("\n[main] training loop finished.")


if __name__ == "__main__":
    main()
