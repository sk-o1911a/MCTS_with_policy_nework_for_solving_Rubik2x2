import os
import torch
import random
from collections import deque
from Policy_Value_Net import PolicyValueNet
from Self_Play import generate_self_play_data
from Train_Network import train_on_selfplay_data
from Plot_Scatter import MetricsLogger
from evaluate import ScrambleEvaluator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


NUM_ITERS = 100
EPISODES_PER_ITER = 80
EPOCHS = 10
BATCH_SIZE = 256
LR = 2e-4
SOLVE_THRESHOLD = 0.9
CHECKPOINT_PATH = "rubik_policy_value.pt"
LOG_DIR = "training_logs"

BUFFER_MAXLEN = 100000
MIN_BUFFER_SIZE = BATCH_SIZE * 8
TRAIN_SAMPLE_SIZE = BATCH_SIZE * 16
replay_buffer = deque(maxlen=BUFFER_MAXLEN)

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
    if scramble_len <= 5:
        return 1
    elif scramble_len <= 9:
        return 0.7
    elif scramble_len <= 11:
        return 0.4
    else:
        return 0.2


device = DEVICE
model = load_or_create_model(device)
logger = MetricsLogger(log_dir=LOG_DIR)
logger.load_json()
evaluator = ScrambleEvaluator()


SCRAMBLE_LEN = 3
recent_solve_rates: list[float] = []

for it in range(1, NUM_ITERS + 1):
    if SCRAMBLE_LEN <= 3:
        SIMULATIONS = 300
    elif SCRAMBLE_LEN <= 7:
        SIMULATIONS = 600
    else:
        SIMULATIONS = 800

    temperature = get_temperature(SCRAMBLE_LEN)

    print(f"\n=== Iteration {it}/{NUM_ITERS} (scramble_len={SCRAMBLE_LEN}, sims={SIMULATIONS}), current temp={temperature} ===")

    dataset, solve_rate = generate_self_play_data(
        model=model,
        num_episodes=EPISODES_PER_ITER,
        max_episode_steps=40,
        num_simulations=SIMULATIONS,
        scramble_len=SCRAMBLE_LEN,
        select_mode="sample",
        temperature=temperature,
        device=device,
    )

    replay_buffer.extend(dataset)

    if len(replay_buffer) < MIN_BUFFER_SIZE:
        print(f"[main] Buffer size ({len(replay_buffer)}) < min size ({MIN_BUFFER_SIZE}). Skipping training.")
        continue

    current_sample_size = min(len(replay_buffer), TRAIN_SAMPLE_SIZE)
    train_dataset = random.sample(replay_buffer, k=current_sample_size)


    recent_solve_rates.append(solve_rate)
    if len(recent_solve_rates) > 8:
        recent_solve_rates.pop(0)

    model, loss, policy_loss, value_loss = train_on_selfplay_data(
        model,
        train_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        device=device,
    )

    logger.log_iteration(
        iteration=it,
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        solve_rate=solve_rate,
        scramble_len=SCRAMBLE_LEN,
        num_samples=len(dataset)
    )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[main] saved checkpoint to {CHECKPOINT_PATH}")
    model.eval()

    if len(recent_solve_rates) == 8:
        avg_solve = sum(recent_solve_rates) / 8.0
        print(f"[main] avg_solve(last 8) = {avg_solve * 100:.1f}%")
        if avg_solve > SOLVE_THRESHOLD and SCRAMBLE_LEN <= 9:
            SCRAMBLE_LEN += 2
            recent_solve_rates.clear()
            print(f"[main] unlock next scramble_len = {SCRAMBLE_LEN}")
        elif avg_solve > SOLVE_THRESHOLD and 9 < SCRAMBLE_LEN <= 15:
            SCRAMBLE_LEN += 1
            recent_solve_rates.clear()
            print(f"[main] unlock next scramble_len = {SCRAMBLE_LEN}")
        else:
            print("[main] stay at current scramble_len")


    logger.save_json()
    logger.plot_all(show=False)


print("\n[main] training loop finished.")

logger.save_json()
evaluator.evaluate_scramble_range()
logger.plot_all(show=True)
evaluator.plot_scramble_results()

