# Rubik's Cube 2x2 Solver using AlphaZero

## Project Information

**Course:** Machine Learning  
**Author:** Nguyen Quoc Khanh  
**Student ID:** 42200211

---

## Overview

This project implements an AI agent that learns to solve a 2x2 Rubik's Cube using reinforcement learning techniques. The system combines Monte Carlo Tree Search (MCTS) with a deep neural network to discover solving strategies through self-play, without any human knowledge or supervision.

## Key Features

- **AlphaZero-inspired Learning**: Self-play reinforcement learning with MCTS
- **Policy-Value Network**: Deep neural network that predicts both move probabilities and position values
- **Action Masking**: Prevents redundant moves (e.g., U followed by U')
- **Progressive Difficulty**: Automatically increases scramble complexity as the agent improves
- **Interactive GUI**: PyGame-based interface for manual control and visualization
- **Curriculum Learning**: Gradually increases problem difficulty from 1-move to 10+ move scrambles

## Project Structure

```
├── Rubik2x2Env.py          # Gymnasium environment for 2x2 Rubik's Cube
├── Policy_Value_Net.py     # Neural network architecture (policy + value heads)
├── MCTS_Core.py            # Monte Carlo Tree Search implementation
├── Action_MCTS.py          # Action selection strategies (greedy/sampling)
├── Self_Play.py            # Self-play data generation
├── Train_Network.py        # Training loop with AlphaZero loss
├── main.py                 # Main training script with curriculum learning
├── test.py                 # Testing script for evaluating trained models
├── PyGame.py               # Interactive GUI for playing and solving
├── Plot_Scatter.py         # Visualization of training progress
└── training_logs           # Save training json logs and plot_scatter pictures 
```

## Technical Details

### Environment

- **State Space**: 6 faces × 2×2 tiles × 6 colors = 144-dimensional one-hot encoding
- **Action Space**: 12 discrete actions (U, U', R, R', F, F', D, D', L, L', B, B')
- **Reward Structure**: +1.0 - 0.05 * steps_used for solved, -1 for unsolved
- **Termination**: Episode ends when cube is solved or max steps reached

### Neural Network Architecture

```
Input (144) 
    ↓
FC(512) + LayerNorm + ReLU
    ↓
FC(1024) + LayerNorm + ReLU
    ↓
FC(512) + LayerNorm + ReLU
    ↓
FC(128) + LayerNorm + ReLU
    ↓
    ├── Policy Head: FC(12) → Softmax
    └── Value Head: FC(1) → Tanh
```

### MCTS Algorithm

1. **Selection**: Traverse tree using UCB formula: `Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))`
2. **Expansion**: Evaluate leaf node with neural network
3. **Backup**: Propagate value estimates back through visited nodes
4. **Action Selection**: Choose action based on visit counts (greedy or temperature-based sampling)

### Training Process

The agent uses curriculum learning with progressive difficulty:

1. Start with 1-move scrambles
2. Generate episodes using MCTS + neural network
3. Train network on self-play data (policy + value loss)
4. When solve rate exceeds 93% for first 4 moves or 90% for subsequent levels, increase scramble length by 1
5. Repeat until reaching target difficulty (maximum 10 moves)

**Hyperparameters:**
- Episodes per iteration: 100
- MCTS simulations: 300-900 (scales with difficulty)
- Training epochs: 8
- Batch size: 256
- Learning rate: 1e-4
- Exploration constant (c_puct): flexible based on scramble length from 1.0 to 0.3

## Installation

### Requirements

- Python 3.8+
- CUDA 12.4 (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/sk-o1911a/MCTS_with_policy_nework_for_solving_Rubik2x2.git
cd cd MCTS_with_policy_nework_for_solving_Rubik2x2

# Install dependencies
pip install -r requirements.txt
```

**Main Dependencies:**
- PyTorch 2.6.0 (with CUDA 12.4 support)
- Gymnasium 1.2.1
- NumPy 2.3.3
- PyGame 2.6.1
- SymPy 1.13.1

## Usage

### Training

Run the main training script with curriculum learning:

```bash
python main.py
```

This will:
- Load or create a new model
- Generate self-play data
- Train the network
- Save checkpoints to `rubik_policy_value.pt`
- Automatically increase difficulty as performance improves

### Testing

Evaluate a trained model on specific scramble lengths:

```bash
python test.py
```

The script will:
- Load the saved model
- Solve a scrambled cube using greedy MCTS
- Display step-by-step solution
- Print ASCII visualization of each state

### Interactive GUI

Launch the PyGame interface:

```bash
python PyGame.py
```

**Controls:**
- **Manual Moves**: Click U, U', R, R', F, F', etc. buttons
- **Scramble**: Set scramble length and click "Scram"
- **Solve**: Click "Solve" to watch AI solve the cube automatically
- **Reset**: Return to solved state

## Results

The agent successfully learns to solve increasingly complex scrambles through self-play:


| Scramble Length | Solve Rate | MCTS Simulations | Temperature |
|-----------------|------------|------------------|-------------|
| 1-3 moves       | ~100%+     | 300              | 1.1         |
| 4-6 moves       | ~99%+      | 700              | 0.6-1.1     |
| 7-10 moves      | ~90%+      | 900              | 0.3-0.6     |


The learned policy demonstrates intelligent move sequences and avoids redundant actions through action masking.

## Algorithm Highlights

### Enhanced Action Masking

Two-layer masking system for efficient search:

1. **Inverse Move Prevention**: Blocks immediately undoing the last move (e.g., R → R' blocked)
2. **Repetition Limit**: Prevents rotating the same face more than 2 consecutive times (e.g., U → U → U blocked)

```python
def _legal_action_mask(self):
    mask = np.ones(12, dtype=bool)
    
    # Block inverse of last action
    if self._last_action is not None:
        mask[self._last_action ^ 1] = False
    
    # Block same face after 2 consecutive rotations
    if self._last_face_count >= 2:
        mask[self._last_face * 2] = False      # CW
        mask[self._last_face * 2 + 1] = False  # CCW
    
    return mask
```

### Self-Play Loop with Curriculum Learning

```python
SCRAMBLE_LEN = 1
recent_solve_rates = []

for iteration in range(NUM_ITERS):
    # 1. Adjust MCTS simulations based on difficulty
    if SCRAMBLE_LEN <= 3:
        SIMULATIONS = 300
    elif SCRAMBLE_LEN <= 6:
        SIMULATIONS = 700
    else:
        SIMULATIONS = 900
    
    # 2. Adjust temperature based on difficulty
    temperature = get_temperature(SCRAMBLE_LEN)
    
    # 3. Generate episodes with adaptive parameters
    dataset, solve_rate = generate_self_play_data(
        model, MCTS, SIMULATIONS, temperature
    )
    
    # 4. Train network
    model = train_on_data(dataset)
    
    # 5. Track and evaluate performance
    recent_solve_rates.append(solve_rate)
    
    # 6. Adjust difficulty when agent masters current level
    if len(recent_solve_rates) == 8:
    avg_solve = sum(recent_solve_rates) / 8.0
    print(f"[main] avg_solve(last 8) = {avg_solve * 100:.1f}%")
    if avg_solve > SOLVE_THRESHOLD + 0.03 and SCRAMBLE_LEN <= 4:
        SCRAMBLE_LEN += 1
        recent_solve_rates.clear()
        print(f"[main] unlock next scramble_len = {SCRAMBLE_LEN}")
    elif avg_solve > SOLVE_THRESHOLD and 4 < SCRAMBLE_LEN < 10:
        SCRAMBLE_LEN += 1
        recent_solve_rates.clear()
        print(f"[main] unlock next scramble_len = {SCRAMBLE_LEN}")
    else:
        print("[main] stay at current scramble_len")
```
### Loss Function
```
Total Loss = MSE(value, target_value) + CrossEntropy(policy, target_policy)
```

Where:
- **Value Loss**: Measures accuracy of position evaluation
- **Policy Loss**: Measures alignment between MCTS visit distribution and policy output

## References

- Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
- Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)
- Gymnasium documentation: https://gymnasium.farama.org/

## License

This project is submitted as coursework for Machine Learning course.

---

**Note:** Training from scratch may take several hours depending on hardware. A GPU is recommended for faster training. Pre-trained checkpoints can be loaded to skip initial training phases.