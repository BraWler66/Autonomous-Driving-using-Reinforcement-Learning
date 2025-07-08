import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

from settings import IM_WIDTH, IM_HEIGHT, SECONDS_PER_EPISODE

# Add CARLA egg to sys.path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarEnv:
    SHOW_CAM = False  # Change to True if you want to display camera views
    STEER_AMT = 1.0

    def __init__(self):
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.front_camera = None
        self.third_person_camera = None

        # Connect to CARLA server
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # Spawn vehicle at a random spawn point
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # Setup front RGB camera
        rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        rgb_cam.set_attribute("fov", "110")

        front_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.front_sensor = self.world.spawn_actor(rgb_cam, front_transform, attach_to=self.vehicle)
        self.actor_list.append(self.front_sensor)
        self.front_sensor.listen(lambda data: self.process_front_img(data))

        # Setup third-person camera (behind and above)
        third_person_transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15))
        self.third_person_sensor = self.world.spawn_actor(rgb_cam, third_person_transform, attach_to=self.vehicle)
        self.actor_list.append(self.third_person_sensor)
        self.third_person_sensor.listen(lambda data: self.process_third_person_img(data))

        # Apply initial control
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)  # Wait for everything to settle

        # Collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, front_transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # Wait for camera data to be ready
        while self.front_camera is None or self.third_person_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        return self.front_camera  # Return initial observation

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_front_img(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = array.reshape((self.im_height, self.im_width, 4))
        img = img[:, :, :3]  # Drop alpha channel
        self.front_camera = img
        self.display_views()

    def process_third_person_img(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = array.reshape((self.im_height, self.im_width, 4))
        img = img[:, :, :3]
        self.third_person_camera = img
        self.display_views()

    def display_views(self):
        if not self.SHOW_CAM:
            return
        if self.front_camera is None or self.third_person_camera is None:
            return

        combined_img = np.hstack((self.front_camera, self.third_person_camera))

        cv2.putText(combined_img, "Front View", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_img, "Third-Person View", (self.im_width + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("CARLA Agent Views", combined_img)
        cv2.waitKey(1)

    def step(self, action):
        # Apply control based on discrete action
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        # Compute speed
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # Determine reward and done
        if len(self.collision_hist) != 0:
            reward = -200
            done = True
        elif kmh < 50:
            reward = -1
            done = False
        else:
            elapsed = time.time() - self.episode_start
            if elapsed > SECONDS_PER_EPISODE:
                reward = 1
                done = True
            else:
                reward = 1
                done = False

        # Return observation, reward, done, info (None here)
        return self.front_camera, reward, done, None

    def destroy_actors(self):
        for actor in self.actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
