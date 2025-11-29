# Rubik's Cube 2×2 Solver using AlphaZero-style Self-Play

## Project Information

- **Course:** Machine Learning  
- **Author:** Nguyen Quoc Khanh  
- **Student ID:** 42200211  

---

## Overview

This project implements an AI agent that learns to solve the 2×2 Rubik’s Cube via **self-play reinforcement learning** with **Monte Carlo Tree Search (MCTS)** and a **policy–value neural network**, inspired by **AlphaZero**.

The agent is not given any human-designed heuristics or solution algorithms. Instead, it learns solving strategies by playing against itself, collecting data from self-play, and training a neural network to guide future search.

The project includes:

- A custom Gymnasium environment for the 2×2 Rubik’s Cube
- A PyTorch policy–value network
- MCTS with action masking
- An AlphaZero-style self-play + training loop
- Curriculum learning (automatically increasing scramble difficulty)
- Evaluation scripts to measure solve rate by scramble length
- A PyGame GUI to interact with and visualize the agent

---

## Project Structure

```text
├── Rubik2x2Env.py          # Rubik 2×2 Gymnasium environment
├── Policy_Value_Net.py     # Policy–value neural network (PyTorch)
├── MCTS_Core.py            # Monte Carlo Tree Search implementation
├── Action_MCTS.py          # Action selection from MCTS visit counts
├── Self_Play.py            # Self-play data generation
├── Train_Network.py        # AlphaZero loss and training loop
├── Plot_Scatter.py         # Metrics logger and training plots
├── evaluate.py             # Evaluate solve rate vs. scramble length
├── main.py                 # Main training script with curriculum learning
├── test.py                 # Single-episode test runner with ASCII output
├── PyGame.py               # PyGame GUI for manual play and AI solving
├── requirements.txt        # Python dependencies
└── training_logs/          # JSON logs, training plots, evaluation plots
```

---

## Technical Details

### Environment: `Rubik2x2Env.py`

- **State Representation**
  - The cube has 6 faces, each 2×2, and 6 colors.
  - Encoded as a **144-dimensional** one-hot vector:
    - 6 faces × 2 × 2 = 24 stickers
    - Each sticker is one-hot over 6 colors → shape (24, 6)
    - Flattened to `(144,)` by `encode_onehot`.

- **Action Space**
  - 12 discrete moves (standard quarter turns):
    - `0: U`, `1: U'`
    - `2: R`, `3: R'`
    - `4: F`, `5: F'`
    - `6: L`, `7: L'`
    - `8: B`, `9: B'`
    - `10: D`, `11: D'`
  - Implemented via `MOVE_FUNCS` and `apply_move_idx`.

- **Scramble Logic**
  - `scramble(cube, k)` applies `k` random moves.
  - Prevents immediately applying a move that is the “other” turn on the same face (e.g. avoids U then U').

- **Action Masking**
  - To avoid wasteful moves, the environment maintains:
    - `self._last_action`
    - `self._last_face`
    - `self._last_face_count`
  - `_legal_action_mask()` returns a boolean mask over actions:
    1. **Inverse move blocking**: after applying action `a`, its inverse `a ^ 1` is masked out.
    2. **Repetition blocking**: if the same face has been turned **2 or more times consecutively**, both actions for that face are masked out (e.g. block both `U` and `U'`).

  ```python
  def _legal_action_mask(self) -> np.ndarray:
      mask = np.ones((self.action_space.n,), dtype=bool)

      # 1. Block immediate inverse of the last action
      if self._last_action is not None:
          inv = self._inverse_move_idx(self._last_action)  # a ^ 1
          if 0 <= inv < self.action_space.n:
              mask[inv] = False

      # 2. Block same-face moves after >= 2 consecutive rotations
      if self._last_face is not None and self._last_face_count >= 2:
          f = self._last_face
          face_a = f * 2
          face_b = f * 2 + 1
          if 0 <= face_a < self.action_space.n:
              mask[face_a] = False
          if 0 <= face_b < self.action_space.n:
              mask[face_b] = False

      return mask
  ```

- **Episode Dynamics**
  - `reset()`:
    - Sets cube to solved.
    - Applies `scramble_len` random moves.
    - Returns observation and (optionally) an `action_mask` in `info`.
  - `step(action)`:
    - Applies the move to the cube.
    - Updates step count and last move statistics.
    - Checks if solved (`terminated`).
    - Checks if `steps >= max_steps` (`truncated`).
    - Returns `(obs, reward, terminated, truncated, info)`:
      - `reward` is always `0.0` here; training uses a terminal value instead.
      - If `use_action_mask=True`, `info["action_mask"]` is provided.

- **ASCII Rendering**
  - `as_ascii()` prints the cube in a simple net format for debugging and `test.py`.

---

### Policy–Value Network: `Policy_Value_Net.py`

- **Input:** state vector of size 144 (float32)
- **Outputs:**
  - **Policy logits:** vector of size 12 (one per action)
  - **Value:** scalar in `[-1, 1]` representing state quality

Architecture:

```text
Input (144)
    ↓
FC(512)  + LayerNorm + ReLU
    ↓
FC(1024) + LayerNorm + ReLU
    ↓
FC(512)  + LayerNorm + ReLU
    ↓
FC(128)  + LayerNorm + ReLU
    ↓
    ├── Policy Head: Linear(128 → 12)
    └── Value Head : Linear(128 → 1) + Tanh
```

The `predict` helper:

```python
@torch.no_grad()
def predict(self, x):
    policy_logits, value = self.forward(x)
    policy = F.softmax(policy_logits, dim=-1)
    return policy.squeeze(0), value.squeeze(0)
```

Used by MCTS to get action priors and state values.

---

### Monte Carlo Tree Search: `MCTS_Core.py` & `Action_MCTS.py`

#### Tree Node: `MCTSNode`

Each node stores:

- `cube`: numpy representation of the cube state
- `obs`: 144-dim observation (one-hot encoding)
- `action_mask`: legal actions in this node
- `parent`: parent node
- `prior`: prior probability $P(s,a)$ from the network
- `children`: dict[action → child_node]
- Visit/value stats:
  - `N`: visit count
  - `W`: total value
  - `Q = W / N`: mean value
- Face repetition info:
  - `last_face`
  - `repeat_count`

#### MCTS Core: `MCTS`

- Constructed with:
  - `model`: `PolicyValueNet`
  - `num_actions`: typically 12
  - `c_puct`: exploration constant
  - `num_simulations`
  - `device`
- Internally uses a `Rubik2x2Env` instance (`mask_env`) to compute action masks at new nodes.

Main operations:

1. **run(root_cube, root_obs, root_action_mask)**

   - Creates a root `MCTSNode`.
   - Repeats for `num_simulations`:
     1. `_select(root)` — traverse tree to a leaf using UCB.
     2. `_expand_and_evaluate(leaf)` — get policy & value from network and create children.
     3. `_backup(path, value, leaf_node)` — update stats along the path and at the leaf.
   - Returns `visit_counts` (float32 array of length `num_actions`).

2. **Selection**

   ```python
   U = self.c_puct * child.prior * np.sqrt(total_N + 1e-8) / (1 + child.N)
   score = child.Q + U
   ```

   - Greedy over `score` for all children not blocked by `action_mask`.

3. **Expansion & Evaluation**

   - If the cube is already solved, returns value = 1.0 and does not expand.
   - Otherwise:
     - Calls `model.predict(obs_t)` to get `(policy, value)`.
     - For each legal action:
       - Applies move to get `new_cube`.
       - Encodes `new_obs`.
       - Syncs `mask_env` with `new_cube` and last-move info.
       - Uses `mask_env._legal_action_mask()` for the child node.

4. **Backup**

   - For each `(node, action)` pair in the path:
     - Updates `N`, `W`, `Q` of the corresponding child.
   - Also updates the leaf node statistics.

#### Action Selection Policy: `Action_MCTS.py`

`pick_action_from_mcts(visit_counts, mode="greedy", temperature=1.0)`:

- `mode="greedy"`:
  - Returns `argmax(visit_counts)`.
- `mode="sample"`:
  - Converts `visit_counts` to a probability distribution:
    - If `sum == 0`, use uniform.
    - Apply temperature:
      - `probs ∝ visit_counts^(1/temperature)`
  - Samples an action from `probs`.

Usage:

- **Training (self-play):** typically `mode="sample"` with a temperature schedule to encourage exploration.
- **Testing/GUI:** usually `mode="greedy"` for deterministic best moves.

---

### Self-Play and Training: `Self_Play.py` & `Train_Network.py`

#### Self-Play Data Generation: `generate_self_play_data`

Arguments:

- `model`: `PolicyValueNet`
- `num_episodes`
- `max_episode_steps`
- `num_simulations`: MCTS simulations per move
- `scramble_len`: scramble difficulty
- `select_mode`: `"sample"` or `"greedy"`
- `temperature`
- `device`

For each episode:

1. Create `Rubik2x2Env(scramble_len=...)`.
2. Reset env to get `obs`, `cube`, `action_mask`.
3. For up to `max_episode_steps`:
   - Run MCTS to get `visit_counts`.
   - Normalize to `pi` (target policy):
     - If `sum == 0`, use uniform.
     - Else `pi = visit_counts / sum`.
   - Store `(obs, pi)` in temporary episode buffers.
   - Select action via `pick_action_from_mcts(visit_counts, mode=select_mode, temperature=temperature)`.
   - Step in the environment and update state.
   - If solved (`terminated`), break.
4. After the episode:
   - If solved:
     - Let `steps_used = step + 1`.
     - Compute terminal value:

       ```python
       z = 1.0 - 0.03 * steps_used
       if z < 0.4:
           z = 0.4
       ```

   - Else:
     - `z = -1.0`.
   - For each state in this episode, append `(state, pi, z)` to the global `dataset`.

Return:

- `dataset`: list of `(state, target_policy, target_value)` tuples.
- `solve_rate = solved_count / num_episodes`.

#### AlphaZero Loss and Training: `Train_Network.py`

**Loss:**

```text
value_loss  = MSE(value_pred, target_v)
policy_loss = CrossEntropy(policy, target_pi)
total_loss  = 1 * value_loss + 1.2 * policy_loss
```

Implementation outline:

```python
def alphazero_loss(policy_logits, value_pred, target_pi, target_v):
    value_pred = value_pred.squeeze(-1)
    value_loss = F.mse_loss(value_pred, target_v)

    log_prob = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_pi * log_prob).sum(dim=-1).mean()

    loss = 1 * value_loss + 1.2 * policy_loss
    return loss, policy_loss, value_loss
```

**Training loop: `train_on_selfplay_data`**

- Converts dataset into tensors.
- Shuffles each epoch.
- Uses `Adam` optimizer with given `lr`.
- Trains for `epochs`, in mini-batches of `batch_size`.
- Prints and returns the final `loss`, `policy_loss`, and `value_loss`.

---

## Main Training Loop with Curriculum: `main.py`

Key hyperparameters:

- `NUM_ITERS = 100`
- `EPISODES_PER_ITER = 80`
- `EPOCHS = 10`
- `BATCH_SIZE = 256`
- `LR = 2e-4`
- `SOLVE_THRESHOLD = 0.9`
- Checkpoint: `rubik_policy_value.pt`
- Log directory: `training_logs`

Replay buffer:

- Max size: `BUFFER_MAXLEN = 100000`
- Minimum buffer size before training: `MIN_BUFFER_SIZE = BATCH_SIZE * 8`
- Max training sample per iteration: `TRAIN_SAMPLE_SIZE = BATCH_SIZE * 16`

### Workflow per iteration

1. **Model setup**
   - `load_or_create_model()` loads from checkpoint if present, else creates a new `PolicyValueNet`.

2. **Difficulty and MCTS simulations**
   - Start with `SCRAMBLE_LEN = 3`.
   - For each iteration:
     - If `SCRAMBLE_LEN ≤ 3`: `SIMULATIONS = 300`
     - Else if `SCRAMBLE_LEN ≤ 7`: `SIMULATIONS = 600`
     - Else: `SIMULATIONS = 800`
   - Temperature schedule via `get_temperature(scramble_len)`:
     - `scramble_len ≤ 5` → `T = 1.0`
     - `≤ 9` → `T = 0.7`
     - `≤ 11` → `T = 0.4`
     - Else → `T = 0.2`

3. **Self-play**
   - Call `generate_self_play_data` with:
     - `num_episodes = EPISODES_PER_ITER`
     - `max_episode_steps = 40`
     - `num_simulations = SIMULATIONS`
     - `scramble_len = SCRAMBLE_LEN`
     - `select_mode = "sample"`
     - `temperature = T`
   - Extend replay buffer with the generated dataset.

4. **Training**
   - If `len(replay_buffer) < MIN_BUFFER_SIZE`: skip training (warmup).
   - Else:
     - Sample `min(len(replay_buffer), TRAIN_SAMPLE_SIZE)` items.
     - Train with `train_on_selfplay_data`, passing:
       - `BATCH_SIZE`, `EPOCHS`, `LR`.
   - Log metrics via `MetricsLogger.log_iteration`.
   - Save checkpoint to `rubik_policy_value.pt`.

5. **Curriculum learning (adjust scramble length)**
   - Track recent solve rates in a sliding window of up to 8 iterations.
   - When there are 8 recent values:
     - Let `avg_solve = mean(recent_solve_rates)`.
     - If `avg_solve > SOLVE_THRESHOLD`:
       - If `SCRAMBLE_LEN ≤ 9`: increase `SCRAMBLE_LEN` by **2**.
       - If `9 < SCRAMBLE_LEN ≤ 15`: increase `SCRAMBLE_LEN` by **1**.
       - Clear `recent_solve_rates`.
     - Else: keep current difficulty.

6. **Logging and visualization**
   - After each iteration:
     - Save metrics JSON (`metrics.json`).
     - Plot training curves (`training_metrics.png`) using `MetricsLogger.plot_all`.
   - After finishing all iterations:
     - Run `ScrambleEvaluator.evaluate_scramble_range()` to gather solve rate vs. scramble length.
     - Plot final training metrics and evaluation bar chart.

---

## Metrics Logging and Plots: `Plot_Scatter.py`

`MetricsLogger` keeps arrays of:

- `iterations`
- `losses`
- `policy_losses`
- `value_losses`
- `solve_rates`
- `scramble_lengths`
- `num_samples`

Functions:

- `save_json("metrics.json")` → `training_logs/metrics.json`
- `load_json("metrics.json")`
- `plot_all("training_metrics.png")`:
  - 2×2 grid:
    - Total loss vs. iteration
    - Solve rate (%) vs. iteration
    - Policy loss vs. iteration
    - Value loss vs. iteration
  - Saves to `training_logs/training_metrics.png`

---

## Evaluation by Scramble Length: `evaluate.py`

`ScrambleEvaluator`:

- Loads a trained model from `rubik_policy_value.pt`.
- `evaluate_scramble_range(min_len=1, max_len=20, num_simulations=1000, max_episode_steps=40)`:
  - For each `scramble_len` in `[min_len, max_len]`:
    - Calls `generate_self_play_data` with:
      - `num_episodes = n_runs`
      - `select_mode = "greedy"` (focus on evaluation, not exploration)
      - `temperature = 0.3`
    - Stores `{"scramble_len": L, "solve_rate": rate}`.
  - Saves list as `training_logs/eval_scramble_result.json`.

- `plot_scramble_results()`:
  - Loads JSON results if needed.
  - Plots bar chart: solve rate (%) vs. scramble length.
  - Saves to `training_logs/eval_bar_chart.png`.

---

## Single-Episode Test: `test.py`

Used for debugging and observing how the agent solves one scrambled cube:

- `load_model(path, device)` loads `PolicyValueNet` from a checkpoint.
- `run_one_episode`:
  - Creates `Rubik2x2Env(scramble_len=...)`.
  - Prints the initial cube via `env.as_ascii()`.
  - For up to `max_steps`:
    - Runs MCTS with `num_simulations`.
    - Selects action via `pick_action_from_mcts(..., mode="greedy")`.
    - Steps the environment and prints:
      - Move name (e.g. `R`) and index.
      - Cube state in ASCII.
    - Stops when solved or out of steps.

---

## PyGame GUI: `PyGame.py`

Provides a graphical interface to:

- Display the 2×2 cube net.
- Apply moves manually with buttons: `U`, `U'`, `R`, `R'`, `F`, `F'`, `L`, `L'`, `B`, `B'`, `D`, `D'`.
- Scramble with a user-specified length via an input box.
- Let the AI solve the current cube automatically using MCTS.
- Animate the solving sequence on screen.

Key function: `solve_with_mcts(env, model, device, num_simulations=200, max_steps=40)`:

- Copies the current cube into a temporary env with `use_action_mask=True`.
- Repeatedly:
  - Runs MCTS.
  - Chooses the best move with `mode="greedy"`.
- Stops if solved or after `max_steps`.
- Returns:
  - `actions`: list of action indices.
  - `formula`: space-separated move sequence (e.g. `"R U R' U'"`).

The main loop:

- Handles UI events.
- Draws cube and buttons.
- Shows the solving formula at the bottom.
- Animates moves of the solution sequence over time.

---

## Installation

### Requirements

- **Python:** 3.8+
- **GPU (recommended):** NVIDIA RTX 5000 series or newer for best performance with the current PyTorch nightly CUDA wheels.

Main dependencies (from `requirements.txt`):

- `torch`, `torchvision`, `torchaudio` (nightly CUDA 12.8 wheels)
- `gymnasium`
- `numpy`
- `pygame`
- `tensorboard`
- `sympy`
- `pillow`

> **Note:**  
> This project uses the PyTorch **nightly** CUDA 12.4 wheels via:
>
> ```text
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```
> 
> If you have an NVIDIA RTX 5000 series or newer GPU, install with:
> 
> ```text
> pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

### Setup

```bash
# Create and activate a virtual environment (optional but recommended)

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Clone the repository
git clone https://github.com/sk-o1911a/MCTS_with_policy_nework_for_solving_Rubik2x2.git
cd MCTS_with_policy_nework_for_solving_Rubik2x2

# Install dependencies (requires RTX 5000 series or newer for the default CUDA 12.8 nightly wheels)
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Agent (Curriculum Learning)

```bash
python main.py
```

This will:

- Load or initialize `rubik_policy_value.pt`.
- For each iteration:
  - Generate self-play data using MCTS + the current network.
  - Append data to the replay buffer.
  - Train the network if the buffer is sufficiently large.
  - Update and log metrics.
  - Save an updated checkpoint.
  - Potentially increase `SCRAMBLE_LEN` when the agent’s recent solve rate exceeds the threshold.

Training logs and plots:

- `training_logs/metrics.json`
- `training_logs/training_metrics.png`

### 2. Evaluate Solve Rate vs. Scramble Length

After you have a trained checkpoint:

```bash
python evaluate.py
```

This will:

- Load the model from `rubik_policy_value.pt`.
- Evaluate solve rate across a range of scramble lengths.
- Save results:
  - `training_logs/eval_scramble_result.json`
  - `training_logs/eval_bar_chart.png`

You can also call the evaluator from inside `main.py` (the script does so at the end of training).

### 3. Run a Single Test Episode

```bash
python test.py
```

- Loads the trained model.
- Scrambles a cube with given `scramble_len`.
- Uses MCTS with greedy selection to solve.
- Prints each move and the cube state in ASCII.
- Reports whether it solved within the step limit.

### 4. Launch the PyGame GUI

```bash
python PyGame.py
```

Controls:

- **Manual Moves:** Click buttons `U`, `U'`, `R`, `R'`, `F`, `F'`, `L`, `L'`, `B`, `B'`, `D`, `D'`.
- **Scramble:**
  - Enter scramble length in the text box.
  - Click **“Scram”** to regenerate a scrambled cube.
- **Solve:**
  - Click **“Solve”**.
  - The AI will compute a solution sequence via MCTS and animate each move.
- **Reset:**
  - Click **“Reset”** to go back to the solved state.

---

## Notes and Limitations

- Training from scratch may take several hours depending on hardware.
- Performance depends on:
  - Number of MCTS simulations per move.
  - Curriculum schedule (how quickly scramble length increases).
  - Model architecture and hyperparameters.
- For faster experimentation:
  - Reduce `NUM_ITERS` or `EPISODES_PER_ITER`.
  - Lower `num_simulations` (at the cost of weaker policy guidance).

---

## References

- Silver, D. et al. (2017). *“Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm”* (AlphaZero).
- Silver, D. et al. (2016). *“Mastering the game of Go with deep neural networks and tree search”* (AlphaGo).
- [Gymnasium documentation](https://gymnasium.farama.org/).

---

**Disclaimer:**  
This project was developed as a course assignment for educational purposes and is not intended for commercial use. If you use or modify the code, please provide appropriate attribution.