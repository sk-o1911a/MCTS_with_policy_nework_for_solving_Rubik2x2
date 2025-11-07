import os
import torch

from Policy_Value_Net import PolicyValueNet
from Self_Play import generate_self_play_data
from Train_Network import train_on_selfplay_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


NUM_ITERS = 200
EPISODES_PER_ITER = 50
EPOCHS = 5
BATCH_SIZE = 256
SIMULATIONS = 0
SCRAMBLE_LEN = 0
LR = 2e-4
SOLVE_THRESHOLD = 0.70
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


def main():
    device = DEVICE
    model = load_or_create_model(device)

    SCRAMBLE_LEN = 1
    recent_solve_rates: list[float] = []

    for it in range(1, NUM_ITERS + 1):
        if SCRAMBLE_LEN <= 3:
            SIMULATIONS = 200
        elif SCRAMBLE_LEN <= 6:
            SIMULATIONS = 300
        else:
            SIMULATIONS = 400

        print(f"\n=== Iteration {it}/{NUM_ITERS} (scramble_len={SCRAMBLE_LEN}, sims={SIMULATIONS}) ===")

        dataset, solve_rate = generate_self_play_data(
            model=model,
            num_episodes=EPISODES_PER_ITER,
            max_episode_steps=50,
            num_simulations=SIMULATIONS,
            scramble_len=SCRAMBLE_LEN,
            select_mode="sample",
            temperature=1.0,
            device=device,
        )
        print(f"[main] self-play collected {len(dataset)} samples.")

        recent_solve_rates.append(solve_rate)
        if len(recent_solve_rates) > 2:
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

        if len(recent_solve_rates) == 2:
            avg_solve = sum(recent_solve_rates) / 2.0
            print(f"[main] avg_solve(last 2) = {avg_solve * 100:.1f}%")
            if avg_solve >= SOLVE_THRESHOLD and SCRAMBLE_LEN < 10:
                SCRAMBLE_LEN += 1
                print(f"[main] unlock next scramble_len = {SCRAMBLE_LEN}")
            else:
                print("[main] stay at current scramble_len")

    print("\n[main] training loop finished.")


if __name__ == "__main__":
    main()
