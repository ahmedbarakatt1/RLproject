# Reinforcement Learning Algorithm Visualizer

A comprehensive interactive web application for exploring, training, and visualizing reinforcement learning algorithms across multiple environments.

## Table of Contents
- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Environments](#environments)
- [Parameter Adjustment Capabilities](#parameter-adjustment-capabilities)
- [Visualization Techniques](#visualization-techniques)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This application provides an interactive platform for understanding and experimenting with fundamental reinforcement learning algorithms. Built with Streamlit, it offers real-time visualization of the learning process, allowing users to observe how agents learn optimal policies through different approaches.

---

## Algorithms Implemented

### 1. Value Iteration (Dynamic Programming)

**Description:** Value Iteration is a model-based algorithm that computes the optimal value function by iteratively applying the Bellman optimality equation until convergence.

**Update Rule:**
$$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$

**Characteristics:**
- ✅ Guaranteed convergence to optimal policy
- ✅ Works with known transition dynamics
- ❌ Requires complete model of the environment
- ❌ Computationally expensive for large state spaces

**Implementation:** [value_iteration.py](value_iteration.py)

---

### 2. Policy Iteration (Dynamic Programming)

**Description:** Policy Iteration alternates between policy evaluation (computing the value function for a given policy) and policy improvement (making the policy greedy with respect to the value function).

**Update Rules:**
- Policy Evaluation: $V^\pi(s) = \sum_{s'} P(s'|s,\pi(s))[R + \gamma V^\pi(s')]$
- Policy Improvement: $\pi(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$

**Characteristics:**
- ✅ Often converges in fewer iterations than Value Iteration
- ✅ Produces optimal policy
- ❌ Each iteration is more computationally expensive
- ❌ Requires complete model of the environment

**Implementation:** [policy_iteration.py](policy_iteration.py)

---

### 3. Monte Carlo Methods

**Description:** Monte Carlo methods learn from complete episodes of experience. They estimate value functions by averaging returns observed after visiting states, without requiring knowledge of the environment's dynamics.

**Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}[G_t - Q(s,a)]$$

Where $G_t$ is the return (cumulative discounted reward) from time $t$.

**Characteristics:**
- ✅ Model-free - learns from experience
- ✅ Unbiased estimates of value functions
- ❌ High variance in estimates
- ❌ Must wait for episode completion to update

**Features:**
- First-visit Monte Carlo implementation
- ε-greedy exploration with decay
- Configurable decay rate and minimum epsilon

**Implementation:** [monte_carlo.py](monte_carlo.py)

---

### 4. Temporal Difference (TD) Learning

**Description:** TD learning combines ideas from Monte Carlo and dynamic programming. It updates estimates based on other estimates (bootstrapping) without waiting for the final outcome.

**Update Rule:**
$$V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]$$

**Characteristics:**
- ✅ Model-free learning
- ✅ Can learn before episode ends (online learning)
- ✅ Lower variance than Monte Carlo
- ❌ Biased estimates due to bootstrapping
- ❌ Sensitive to learning rate selection

**Implementation:** [td.py](td.py)

---

### 5. SARSA (On-Policy TD Control)

**Description:** SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm that learns Q-values while following an ε-greedy policy. It updates based on the action actually taken.

**Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma Q(s',a') - Q(s,a)]$$

**Characteristics:**
- ✅ On-policy - learns about the policy being followed
- ✅ Safer exploration in dangerous environments (e.g., cliff walking)
- ❌ May not find optimal policy if exploration continues
- ❌ Slower convergence compared to Q-Learning

**Features:**
- Terminal state handling for proper value propagation
- ε-greedy exploration with configurable decay

**Implementation:** [sarsa.py](sarsa.py)

---

### 6. Q-Learning (Off-Policy TD Control)

**Description:** Q-Learning is an off-policy TD control algorithm that learns the optimal Q-function regardless of the policy being followed. It always updates using the maximum Q-value of the next state.

**Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**Characteristics:**
- ✅ Off-policy - can learn optimal policy while exploring
- ✅ Simple and widely applicable
- ✅ Faster convergence in many scenarios
- ❌ Can overestimate Q-values
- ❌ May be unstable with function approximation

**Features:**
- Terminal state handling (no bootstrapping at terminal states)
- Epsilon decay for exploration-exploitation balance

**Implementation:** [q_learning.py](q_learning.py)

---

## Environments

### 1. FrozenLake-v1

**Description:** Navigate a frozen lake from start (S) to goal (G) while avoiding holes (H). The agent receives a reward of +1 only upon reaching the goal.

**State Space:** 16 discrete states (4×4 grid)
**Action Space:** 4 actions (Left, Down, Right, Up)

**Special Features:**
- **Slippery Mode:** Optional stochastic transitions where the agent may slip and move in unintended directions
- **Sparse Rewards:** Only +1 at goal, making learning challenging

**Grid Layout:**
```
S F F F
F H F H
F F F H
H F F G
```

---

### 2. Taxi-v3

**Description:** A taxi navigation task where the agent must pick up a passenger from one location and drop them off at another destination.

**State Space:** 500 discrete states (5×5 grid × 5 passenger locations × 4 destinations)
**Action Space:** 6 actions (South, North, East, West, Pickup, Dropoff)

**Rewards:**
- +20 for successful dropoff
- -1 per step (encourages efficiency)
- -10 for illegal pickup/dropoff attempts

---

### 3. CliffWalking-v1

**Description:** Navigate from start to goal along a cliff edge. Falling off the cliff results in a large negative reward and returns the agent to the start.

**State Space:** 48 discrete states (4×12 grid)
**Action Space:** 4 actions (Up, Right, Down, Left)

**Rewards:**
- -1 per step
- -100 for falling off the cliff
- Episode terminates upon reaching the goal

**Notable:** This environment clearly demonstrates the difference between SARSA (safer, longer path) and Q-Learning (optimal but riskier path).

---

## Parameter Adjustment Capabilities

The application provides extensive parameter tuning through an interactive sidebar:

### Training Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Discount Factor (γ)** | 0.0 - 1.0 | 0.99 | Controls importance of future rewards. Higher values prioritize long-term rewards. |
| **Number of Iterations** | 10 - 1000 | 100 | For Value/Policy Iteration: maximum iterations for convergence. |
| **Number of Episodes** | 100 - 50000 | 10000 | For model-free algorithms: total training episodes. |

### Exploration Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Initial Epsilon (ε)** | 0.0 - 1.0 | 1.0 | Starting exploration rate for ε-greedy policy. |
| **Epsilon Decay Rate** | 0.9 - 0.9999 | 0.9995 | Multiplicative decay applied after each episode. |
| **Minimum Epsilon** | 0.0 - 0.5 | 0.01 | Floor value to maintain some exploration. |

### Learning Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Learning Rate (α)** | 0.01 - 1.0 | 0.1 | Step size for TD updates. Higher = faster but potentially unstable. |

### Environment-Specific Options

| Option | Environment | Description |
|--------|-------------|-------------|
| **Slippery Ice** | FrozenLake | Enables stochastic transitions (33% chance of slipping) |

---

## Visualization Techniques

### 1. Value Function Heatmap

**Purpose:** Displays the estimated value of each state, showing which states are most desirable.

**Features:**
- Color gradient from red (low value) through yellow to green (high value)
- Numerical values displayed in each cell
- Adaptive font sizing for different grid sizes (smaller for CliffWalking's 4×12 grid)
- Special handling for Taxi's large state space (shows representative subset)

**Interpretation:** States closer to the goal typically have higher values, with the gradient showing how value propagates through the state space.

---

### 2. Policy Visualization

**Purpose:** Shows the optimal action to take in each state using directional arrows.

**Features:**
- Arrow symbols indicating recommended action (←, →, ↑, ↓)
- Grid-based layout matching environment structure
- Clear visual representation of the learned strategy

**Interpretation:** Following the arrows from any state should lead to the goal via the optimal path.

---

### 3. Training Convergence Plot

**Purpose:** Tracks learning progress over training episodes.

**Metrics Displayed:**
- **Episode Rewards:** Total reward accumulated per episode
- **Exploration Rate:** Epsilon decay over time (for applicable algorithms)

**Features:**
- Line charts showing trends over episodes
- Helps identify:
  - Learning speed
  - Convergence stability
  - Exploration-exploitation balance

---

### 4. Agent Inference (Policy Execution)

**Purpose:** Demonstrates the trained agent navigating the environment.

**Features:**
- **Animated Playback:** Watch the agent execute its learned policy step-by-step
- **Adjustable Speed:** Control animation speed (0.1s - 2.0s per step)
- **Frame-by-Frame Navigation:** Manual slider for detailed inspection
- **Step Details:** Shows state, action, and reward at each step
- **Trajectory Summary:** 
  - Total steps taken
  - Total reward accumulated
  - Goal reached indicator

**Information Displayed Per Step:**
- Current state number
- Action taken (with descriptive label)
- Immediate reward received
- Visual rendering of the environment

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RLApplication
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `streamlit` - Web application framework
- `gymnasium` - RL environments
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `pandas` - Data handling for charts

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Workflow

1. **Select Environment:** Choose from FrozenLake, Taxi, or CliffWalking
2. **Configure Parameters:** Adjust algorithm-specific parameters in the sidebar
3. **Choose Algorithm:** Select the RL algorithm to train
4. **Train Agent:** Click "Train Agent" to start training
5. **View Results:** Explore the value function, policy, and convergence plots
6. **Run Inference:** Watch the trained agent navigate the environment

### Tips for Best Results

- **FrozenLake:** Use 10,000+ episodes for model-free algorithms
- **CliffWalking:** Compare SARSA vs Q-Learning to see on-policy vs off-policy differences
- **Taxi:** Requires more episodes due to larger state space (500 states)
- **Slow Epsilon Decay:** Use 0.9995+ for sparse reward environments

---

## Project Structure

```
RLApplication/
├── app.py                 # Main Streamlit application
├── value_iteration.py     # Value Iteration algorithm
├── policy_iteration.py    # Policy Iteration algorithm
├── monte_carlo.py         # Monte Carlo methods
├── td.py                  # Temporal Difference learning
├── sarsa.py              # SARSA algorithm
├── q_learning.py         # Q-Learning algorithm
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- OpenAI Gymnasium Documentation: https://gymnasium.farama.org/
- Streamlit Documentation: https://docs.streamlit.io/

---

## License

This project is for educational purposes as part of a Reinforcement Learning course.