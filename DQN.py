import random
import time
from collections import deque

import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from settings import (
    IM_WIDTH, IM_HEIGHT, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE,
    MINIBATCH_SIZE, PREDICTION_BATCH_SIZE, TRAINING_BATCH_SIZE,
    UPDATE_TARGET_EVERY, MODEL_NAME, DISCOUNT, AGGREGATE_STATS_EVERY
)
from modified_tensorboard import ModifiedTensorBoard


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.terminate = False
        self.training_initialized = False
        self.step = 1

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        output = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255.0
        current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE, verbose=0)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255.0
        future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE, verbose=0)

        X, y = [], []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = self.step % AGGREGATE_STATS_EVERY == 0
        self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0,
                       callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
        self.step += 1

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.0, verbose=0)[0]

    def train_in_loop(self):
        # Warm-up run to avoid TensorFlow threading issues
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        self.model.fit(X, y, verbose=False, batch_size=1)
        self.training_initialized = True

        while not self.terminate:
            self.train()
            time.sleep(0.01)
