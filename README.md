# 🚗 Autonomous-Driving-using-Reinforcement-Learning


A Deep Q-Learning (DQN) based autonomous driving agent trained in the CARLA Simulator. The agent learns to navigate a simulated urban environment using reinforcement learning techniques.

---

## 📂 Project Structure

````

carla-rl-autonomous-car/
│
├── **pycache**/         # Python cache files
├── logs/                # Training logs and models
├── models/              # Saved DQN model checkpoints
│
├── CarUI.py             # Manual driving & UI interface
├── DQN.py               # DQN model and agent logic
├── main.py              # Main training/evaluation script
├── settings.py          # Hyperparameters and config
├── Tensorboard.py       # TensorBoard logging
├── Visual.py            # Reward plotting and visualization
└── README.md            # Project documentation

````

---

## 🚀 Features

- 🧠 Deep Q-Learning with experience replay & target network
- 🌆 Integration with CARLA simulator (urban driving)
- 📊 TensorBoard for training visualization
- 🕹 Manual driving support for testing and debugging

---

## 🛠 Installation

### Step 1: Clone and Set Up

```bash
git clone https://github.com/yourusername/carla-rl-autonomous-car.git
cd carla-rl-autonomous-car

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

pip install -r requirements.txt
````

### Step 2: Install CARLA

Download CARLA (recommended: 0.9.13 or later) from:
[https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

Follow their installation instructions for your OS.

---

## 🏃 Usage

### Train the Agent

```bash
python main.py --train
```

### Evaluate a Trained Model

```bash
python main.py --evaluate --model models/dqn_latest.pth
```

### Visualize Training Results

```bash
python Visual.py
```

---

## ⚙ Configuration

````
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
MEMORY_FRACTION = 0.4
MIN_REWARD = -200
EPISODES = 20
DISCOUNT = 0.99
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
````
---

## 📊 Monitor Training

Launch TensorBoard:

```bash
python Tensorboard.py
# or
tensorboard --logdir logs/
```


## 📦 Requirements

* Python 3.8+
* PyTorch
* NumPy
* OpenCV
* TensorBoard
* CARLA Python API

## Group Members
👥
- **Muhammad Taqui**
- **Babar Ali**
