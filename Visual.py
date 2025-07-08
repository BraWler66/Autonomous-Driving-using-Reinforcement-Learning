import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from final_code import MEMORY_FRACTION
from CarUI import CarEnv

MODEL_PATH = r'C:\Users\Babar\Desktop\WindowsNoEditor\PythonAPI\examples\models\Xception__final__1748310079.model'

if __name__ == '__main__':

    # Configure GPU memory fraction for TensorFlow 2.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            mem_limit_mb = int(1024 * MEMORY_FRACTION)  # Adjust if you want another base memory size
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit_mb)]
                )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
        except RuntimeError as e:
            print(e)

    # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - use the shape expected by the model (480x640x3)
    model.predict(np.ones((1, 480, 640, 3)))

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            cv2.imshow('Agent - preview', current_state)
            cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape) / 255)[0]
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter) / sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')

        # Destroy all actors at end of episode
        for actor in env.actor_list:
            actor.destroy()
