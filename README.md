
# 🚗 Autonomous Driving using Deep Q-Learning (CARLA Simulator)

A Deep Q-Network (DQN) based autonomous driving agent trained in the CARLA simulator. The agent learns to drive in an urban environment using reinforcement learning.

## 📁 Project Structure

```

carla-rl-autonomous-car/
│
├── *pycache*/            # Python cache files
├── logs/                 # Training logs and TensorBoard data
├── models/               # Saved DQN model checkpoints
│
├── CarUI.py              # Manual driving UI using CARLA
├── DQN.py                # Deep Q-Network model & logic
├── main.py               # Main script for training/evaluation
├── settings.py           # Configuration and hyperparameters
├── Tensorboard.py        # TensorBoard launcher (optional)
├── Visual.py             # Training reward plots
├── LICENSE
└── README.md

````

---

## 🚀 Features

- 🧠 Deep Q-Learning with experience replay and target network
- 🌆 Urban driving using the CARLA simulator
- 📈 Live monitoring with TensorBoard
- 🕹 Manual driving interface for testing/debugging

---

## 🛠 Installation

### Step 1: Clone and Set Up Environment

```bash
git clone https://github.com/yourusername/carla-rl-autonomous-car.git
cd carla-rl-autonomous-car

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install required packages
pip install torch torchvision numpy opencv-python tensorboard
````

### Step 2: Install CARLA

Download CARLA (recommended: version 0.9.13 or later):

🔗 [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

Follow the installation steps provided for your OS.

---

## 🧪 How to Run

### ▶️ Train the Agent

```bash
python main.py
```

*(Make sure training mode is enabled inside `main.py`)*

---

### 📊 Evaluate a Trained Model

Make sure to load your trained model path inside the script and run:

```bash
python main.py
```

---

### 📈 Visualize Rewards

```bash
python Visual.py
```

---

### 📉 Monitor with TensorBoard

```bash
python Tensorboard.py
# or
tensorboard --logdir logs/
```

---

## ⚙ Sample Configuration (`settings.py`)

```python
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
DISCOUNT = 0.99
EPISODES = 20
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
```

---

## ✅ Requirements

* Python 3.8+
* PyTorch
* NumPy
* OpenCV
* TensorBoard
* CARLA Python API

---

## 👥 Group Members

* **Muhammad Taqui**
* **Babar Ali**

```
