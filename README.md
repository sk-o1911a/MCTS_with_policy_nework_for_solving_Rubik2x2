# Rubik's Cube 2x2 Solver using AlphaZero

## Project Information

**Course:** Machine Learning  
**Author:** Nguyễn Quốc Khánh  
**Student ID:** 42200211

---

## Overview

This project implements an AI agent that learns to solve a 2x2 Rubik's Cube using reinforcement learning techniques inspired by AlphaZero. The system combines Monte Carlo Tree Search (MCTS) with a deep neural network to discover solving strategies through self-play, without any human knowledge or supervision.

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
├── Action_MTCS.py          # Action selection strategies (greedy/sampling)
├── Self_Play.py            # Self-play data generation
├── Train_Network.py        # Training loop with AlphaZero loss
├── main.py                 # Main training script with curriculum learning
├── test.py                 # Testing script for evaluating trained models
└── PyGame.py               # Interactive GUI for playing and solving
```

## Technical Details

### Environment

- **State Space**: 6 faces × 2×2 tiles × 6 colors = 144-dimensional one-hot encoding
- **Action Space**: 12 discrete actions (U, U', R, R', F, F', D, D', L, L', B, B')
- **Reward Structure**: Binary (+1 for solved, -1 for unsolved)
- **Termination**: Episode ends when cube is solved or max steps reached

### Neural Network Architecture

```
Input (144) 
    ↓
FC(512) + LayerNorm + ReLU
    ↓
FC(256) + LayerNorm + ReLU
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
4. When solve rate exceeds 70% for 2 consecutive iterations, increase scramble length
5. Repeat until reaching target difficulty (10+ moves)

**Hyperparameters:**
- Episodes per iteration: 50
- MCTS simulations: 200-400 (scales with difficulty)
- Training epochs: 5
- Batch size: 256
- Learning rate: 2e-4
- Exploration constant (c_puct): 1.5

## Installation

### Requirements

- Python 3.8+
- CUDA 12.4 (for GPU support)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <project-folder>

# Install dependencies
pip install -r requirements.txt
```

**Main Dependencies:**
- PyTorch 2.6.0 (with CUDA 12.4 support)
- Gymnasium 1.2.1
- NumPy 2.3.3
- PyGame 2.6.1
- TensorBoard 2.20.0 (for training visualization)
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

- **1-3 moves**: ~95%+ solve rate with 200 simulations
- **4-6 moves**: ~80%+ solve rate with 300 simulations
- **7-10 moves**: ~60%+ solve rate with 400 simulations

The learned policy demonstrates intelligent move sequences and avoids redundant actions through action masking.

## Algorithm Highlights

### Action Masking
Prevents the agent from making inverse moves (e.g., R followed by R'), reducing the effective branching factor and improving search efficiency.

### Self-Play Loop
```python
for iteration in range(NUM_ITERS):
    # 1. Generate episodes
    dataset = generate_self_play_data(model, MCTS)
    
    # 2. Train network
    model = train_on_data(dataset)
    
    # 3. Evaluate & adjust difficulty
    if solve_rate > threshold:
        scramble_len += 1
```

### Loss Function
```
Total Loss = MSE(value, target_value) + CrossEntropy(policy, target_policy)
```

## Future Improvements

- Implement value network bootstrapping for longer horizons
- Add experience replay buffer for more stable training
- Experiment with larger network architectures (ResNets, Transformers)
- Extend to 3×3 Rubik's Cube
- Implement parallel MCTS for faster training
- Add model-based planning with learned dynamics

## References

- Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
- Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)
- Gymnasium documentation: https://gymnasium.farama.org/

## License

This project is submitted as coursework for Machine Learning course.

---

**Note:** Training from scratch may take several hours depending on hardware. A GPU is recommended for faster training. Pre-trained checkpoints can be loaded to skip initial training phases.