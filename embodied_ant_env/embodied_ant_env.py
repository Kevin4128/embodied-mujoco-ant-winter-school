import numpy as np
import threading
import time
from collections import defaultdict

from imu_msp import IMU_MSP
from motor_controller import MotorController
from apriltag_tracking import VisionTracker, show_image
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box

class GoToTargetTask:
    def __init__(self, target_position=np.array([1.0, 0.0]), action_cost_weight=0.01, success_radius=0.15):
        self.target_position = target_position
        self.action_cost_weight = action_cost_weight
        self.success_radius = success_radius
        self.last_pos = None
        self.last_action = np.zeros(8)
        self.last_distance = None
        
        # We need a larger observation space (26) than ForwardTask (24)
        # to include the relative X and Y distance to the target.
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(26,), dtype=np.float32)

    def reset(self, info, action=np.zeros(8)):
        self.last_pos = None
        self.last_distance = None
        self.last_action = np.zeros(8)
        # You can optionally randomize self.target_position here if you want general navigation
        return self(info, action)

    def _make_observation(self, info, target_pos):
        # 1. Get current position and heading
        pos = np.array([info['current_x_position'], info['current_y_position']])
        hx, hy = info['heading_vector'] 
        
        # 2. Calculate the vector pointing to the target in the WORLD frame
        rel_world = target_pos - pos
        
        # 3. Project that vector into the ROBOT'S local frame (Forward and Lateral)
        # This is crucial: it tells the robot "target is 2m ahead" or "target is 1m to the left"
        rel_robot_x = rel_world[0] * hx + rel_world[1] * hy
        rel_robot_y = -rel_world[0] * hy + rel_world[1] * hx
        
        # 4. Construct the full observation vector
        return np.concatenate([
            [rel_robot_x, rel_robot_y], # The new "Go To" info
            info['joint_positions'],
            info['joint_velocities'],
            info['heading_vector'],
            [info['ax'], info['ay'], info['az'], info['wx'], info['wy'], info['wz']],
        ], axis=None).astype(np.float32)

    def __call__(self, info, action):
        pos = np.array([info['current_x_position'], info['current_y_position']])
        target = self.target_position
        distance = np.linalg.norm(pos - target)
        
        # Reward Logic:
        reward = 0.0
        
        # 1. Shaping reward: Give points for getting closer compared to the last step
        if self.last_distance is not None:
            reward += (self.last_distance - distance) * 40.0 

        # 2. Penalty for jerky movement
        cost_action = np.sum(np.square(self.last_action - action)) * self.action_cost_weight
        reward -= cost_action

        self.last_distance = distance
        self.last_action = action.copy()
        
        # 3. Check for success (did we reach the target?)
        terminated = distance < self.success_radius
        if terminated: 
            reward += 10.0 # Bonus for reaching goal
            print(f"Target Reached! {pos}")
        
        observation = self._make_observation(info, target)
        return observation, reward, terminated, False
    
class ForwardTask:
    def __init__(self, action_cost_weight=0.0):
        self.action_cost_weight = action_cost_weight
        self.last_pos = None
        self.last_action = np.zeros(8)
        self.reward_direction_I = np.array([1, 0])
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(24,), dtype=np.float32)
        self.previous_pos_timestamp = None
        print('ForwardTask initialized!')

    def reset(self, info, action=np.zeros(8)):
        self.last_pos = None
        self.previous_pos_timestamp = None
        return self(info, action)

    def __call__(self, info, action):
        pos = np.array([info['current_x_position'], info['current_y_position']])
        pos_timestamp = info['position_timestamp']
        if self.last_pos is None:
            self.last_pos = pos
            progress = 0.0
        else:
            progress = (pos - self.last_pos)[0]

        if self.previous_pos_timestamp is None:
            self.previous_pos_timestamp = pos_timestamp

        #if pos_timestamp == self.previous_pos_timestamp:
            #print('Warning!! The frames did not update!')

        cost_action = np.sum(np.square(self.last_action - action)) * self.action_cost_weight
        self.last_pos = pos
        self.previous_pos_timestamp = pos_timestamp
        self.last_action = action.copy()
        terminated = False
        truncated = False

        reward = progress - cost_action
        info['reward_direction_I'] = self.reward_direction_I
        info['original_reward'] = reward
        info['actions'] = action
        observation = np.concatenate([
            info['joint_positions'],
            info['joint_velocities'],
            info['heading_vector'],
            info['ax'],
            info['ay'],
            info['az'],
            info['wx'],
            info['wy'],
            info['wz'],
        ], axis=None)
        return observation, reward, terminated, truncated

class EmbodiedAnt(gym.Env):

    def __init__(self, motor_controller, imu, tracker, dt=0.02, render_mode=None, joint_config=None, task=ForwardTask()):
        super().__init__()
        self.task = task
        self.motor_controller = motor_controller
        self.motor_controller.enable()
        self.dt = dt
        self.last_step_time = None
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.vis_frame = None
        self.i = 0
        if joint_config is None:
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(45),
                'knee_range': np.radians(20),
            }
        self.joint_config = joint_config

        self._threads_should_exit = False

        self.observation_space = task.observation_space

        self.action_space = Box(
            low=-1, high=1, shape=(8,), dtype=np.float64
        )

        self.imu = imu
        self._imu_data = None
        self._imu_data_lock = threading.Lock()
        self._imu_thread = threading.Thread(target=self._poll_imu, daemon=True)
        self._imu_thread.start()

        self.tracker = tracker
        self._tracker_data = None
        self._tracker_data_lock = threading.Lock()
        self._tracker_thread = threading.Thread(target=self._poll_tracker, daemon=True)
        self._tracker_thread.start()

        self.last_pos = None
        self.last_heading_vector = None
        self.last_seen = 0
        self.last_position_timestamp = None

        self.temperature_log = open('temperature_log.csv', 'a')
        # self.temperature_log = open('temperature_log.csv', 'w')
        self.error_log = open('error_log.csv', 'w')

    def __del__(self):
        self.close()

    def reset(self, seed=None, options=None):
        self.step(np.zeros(self.action_space.shape[0]))
        print('reset(): please move the ant back to the origin.')
        user_input = input('press enter when ready')
        self.last_step_time = time.time()
        info = self.get_observation()
        observation, reward, terminated, truncated = self.task.reset(info)
        return observation, info

    def step(self, action, sleep_until_next_step=True):
        if self._threads_should_exit:
            raise RuntimeError("EmbodiedAnt.step() called after close()")

        # Apply action.
        action = action.copy()
        for i in range(4):
            action[2*i] = np.clip(action[2*i], -1, 1) * self.joint_config['hip_range'] + self.joint_config['hip_zero']
            action[2*i + 1] = np.clip(action[2*i + 1], -1, 1) * self.joint_config['knee_range'] + self.joint_config['knee_zero']
        self.motor_controller.set_positions(action)

        # Sleep.
        sleep_duration = self.dt
        time_since_last_step = 0.0
        if self.last_step_time is not None:
            time_since_last_step = time.time() - self.last_step_time
            sleep_duration = self.dt - time_since_last_step
            if sleep_duration < 0:
                print(f"Warning: calls to step() exceeded step size (time since last step: {time_since_last_step:.3f}s).")
                sleep_duration = 0
        if sleep_until_next_step:
            time.sleep(sleep_duration)
        self.last_step_time = time.time()

        # # Sleep.
        # time.sleep(0.02*self.dt)

        # Get observation.
        time_start = time.time()
        info = self.get_observation()
        info['env_time_to_get_obs'] = time.time() - time_start

        # Get reward and task termination.
        observation, reward, terminated, truncated = self.task(info, action)
        info['env_sleep_duration'] = sleep_duration + time_since_last_step

        # self.temperature_log.write(f"{time.time()}, " + ", ".join(map(str, info['temperatures'])) + "\n")
        # self.temperature_log.flush()

        time_start = time.time()
        errors = self.motor_controller.check_errors()
        if len(errors) > 0: # only log errors if there are any
            self.error_log.write(f"{time.time()}, " + ", ".join(map(str, errors)) + "\n")
            self.error_log.flush()
        info['env_time_to_check_errors'] = time.time() - time_start

        time_start = time.time()
        if len(errors) > 0:
            print('motor controller errors:')
            for error in errors:
                print(error[2])
            truncated = True
            self.motor_controller.recover_from_error()
        info['env_time_to_recover_motor_errors'] = time.time() - time_start

        if self.tracker_lost(info):
            truncated = True

        if self.render_mode == 'human':
            self.i += 1
            if self.i % 10 == 0:
                show_image(self.vis_frame)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human' or self.render_mode == 'rgb_array':
            return self.vis_frame
        return None

    def get_observation(self):
        # IMU.
        time_start = time.time()
        with self._imu_data_lock:
            if self._imu_data is not None:
                imu_data = self._imu_data.copy()
            else:
                imu_data = defaultdict(lambda: 0)
        info = imu_data
        info['env_time_imu_thread'] = time.time() - time_start

        # Tracker.
        time_start = time.time()
        with self._tracker_data_lock:
            if self._tracker_data is not None:
                bodies, frame, vis_frame = self._tracker_data
            else:
                bodies, frame, vis_frame = {}, np.zeros((640, 480, 3)), np.zeros((640, 480, 3))
        info['bodies'] = bodies
        if 'body' in bodies:
            info['current_x_position'] = bodies['body']['position'][0]
            info['current_y_position'] = bodies['body']['position'][1]
            self.last_pos = bodies['body']['position']
            self.last_seen = time.time()
            heading_vector = (bodies['body']['orientation'] @ np.array([1, 0, 0]))[:2]
            heading_vector /= np.linalg.norm(heading_vector)
            self.last_heading_vector = heading_vector
            info['position_timestamp'] = bodies['body']['timestamp']
            self.last_position_timestamp = info['position_timestamp']
            info['detection_time'] = bodies['body']['detection_time']
        else:
            info['current_x_position'] = self.last_pos[0] if self.last_pos is not None else 0.0
            info['current_y_position'] = self.last_pos[1] if self.last_pos is not None else 0.0
            heading_vector = self.last_heading_vector
            info['position_timestamp'] = self.last_position_timestamp if self.last_position_timestamp is not None else 0.0
            info['detection_time'] = 0.0
        info['heading_vector'] = heading_vector

        self.vis_frame = vis_frame
        info['env_time_tracker_thread'] = time.time() - time_start

        # Motor outputs.
        time_start_motor_feedback = time.time()
        joint_positions, joint_velocities, joint_loads = self.motor_controller.get_feedback()
        #print('time to get motor feedback: ', time.time() - time_start_motor_feedback)
        # temperatures = self.motor_controller.get_temperature()
        info['joint_positions'] = joint_positions
        info['joint_velocities'] = joint_velocities
        info['joint_loads'] = joint_loads
        # info['temperatures'] = temperatures
        info['env_time_get_motor_feedback'] = time.time() - time_start

        return info

    def tracker_lost(self, info):
        if time.time() - self.last_seen > 2:
            # print('body tracker not seen for 2 seconds')
            return True
        if 'body' in info['bodies']:
            img_pos = info['bodies']['body']['image_pos']
            if img_pos[0] < 0.1 or img_pos[0] > 0.9 or img_pos[1] < 0.1 or img_pos[1] > 0.9:
                print('body is out of camera frame')
                return True # body is out of frame
        return False

    def close(self):
        self._threads_should_exit = True
        self._imu_thread.join()
        self.motor_controller.disable()

    def _poll_imu(self):
        while not self._threads_should_exit:
            try:
                imu_data = self.imu.get_data()
                with self._imu_data_lock:
                    self._imu_data = imu_data
            except Exception as e:
                print(f"Error in _poll_imu: {e}")
                self._threads_should_exit = True

    def _poll_tracker(self):
        while not self._threads_should_exit:
            try:
                data = self.tracker.track()
                with self._tracker_data_lock:
                    self._tracker_data = data
            except Exception as e:
                print(f"Error in _poll_tracker: {e}")
                self._threads_should_exit = True

def make_ant_env(cfg, **kwargs):
    motor_controller = MotorController(port=cfg['motor_port'], motor_list=cfg['motor_list'])
    imu = IMU_MSP(port=cfg['imu_port'])
    tracker = VisionTracker(camera_id=cfg['camera_id'],
                            fov_diagonal_deg=cfg['camera_fov_diagonal_deg'],
                            tag_sizes=cfg['camera_tag_sizes'],
                            tag_ids=cfg['camera_tag_ids'])
    return EmbodiedAnt(motor_controller=motor_controller, imu=imu, tracker=tracker, **kwargs)

class DummyMotorController:
    def __init__(self, port=None, motor_list=[0]*8):
        self.nb_motors = len(motor_list)
    def set_positions(self, positions):
        pass
    def get_feedback(self):
        return np.zeros(self.nb_motors), np.zeros(self.nb_motors), np.zeros(self.nb_motors)
    def disable(self):
        pass
    def enable(self):
        pass
    def get_temperature(self):
        return np.zeros(self.nb_motors)
    def check_errors(self):
        return []
    def recover_from_error(self):
        pass

class DummyIMU:
    def __init__(self, port=None):
        pass
    def get_data(self):
        return {'ax': 0, 'ay': 0, 'az': 9.81,
                'wx': 0, 'wy': 0, 'wz': 0,
                'mx': 0, 'my': 0, 'mz': 0,
                'roll_deg': 0, 'pitch_deg': 0, 'yaw_deg': 0,
                'timestamp': time.time()}

class DummyTracker:
    def __init__(self, detector=None, inertial_tag_id=None):
        pass
    def track(self):
        return {}, np.zeros((640, 480, 3)), np.zeros((640, 480, 3))


if __name__ == "__main__":
    import sys
    import json
    cfg = json.load(open(sys.argv[1]))
    motor_controller = MotorController(port=cfg['motor_port'], motor_list=cfg['motor_list'])
    # motor_controller = DummyMotorController()
    imu = IMU_MSP(port=cfg['imu_port'])
    # imu = DummyIMU()
    print(cfg)
    tracker = VisionTracker(camera_id=cfg['camera_id'],
                            fov_diagonal_deg=cfg['camera_fov_diagonal_deg'],
                            tag_sizes=cfg['camera_tag_sizes'],
                            tag_ids=cfg['camera_tag_ids'])
    env = EmbodiedAnt(motor_controller=motor_controller, imu=imu, tracker=tracker, dt=0.05)
    i = 0

    # Go: max, min, max, min (for each motor)
    action_high = np.ones(env.action_space.shape)
    action_low = -np.ones(env.action_space.shape)
    actions = [action_high, action_low]
    step_idx = 0
    while True:
        time_now = time.time()
        # Alternate: max, min, max, min on all motors
        # action = actions[step_idx % 2]
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        # time.sleep(0.20)
        step_idx += 1
        #     show_image(info['vis_frame'])
