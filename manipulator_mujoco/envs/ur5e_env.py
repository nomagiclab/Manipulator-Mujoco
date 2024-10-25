import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.utils.transform_utils import quat2euler, euler2quat


class UR5eEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=255.1, shape=(4,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################

        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.mjcf_model)

        # ur5e arm
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                "../assets/robots/ur5e/ur5e.xml",
            ),
            eef_site_name="eef_site",
            attachment_site_name="attachment_site",
        )

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0, 0, 0], quat=[0.7071068, 0, 0, -0.7071068]
        )

        self._box = Primitive(
            type="box",
            size=[0.02, 0.02, 0.02],
            pos=[0, 0, 0.02],
            rgba=[1, 0, 0, 1],
            friction=[1, 0.3, 0.0001],
        )

        # attach box to arena as free joint
        # TODO: set random position
        self._random_box_pos = [0.5, 0, 0.2]
        self._arena.attach_free(
            self._box.mjcf_model,
            pos=[self._random_box_pos[0], self._random_box_pos[1], 0],
        )
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints[:6],
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        self._griper_actuator = self._physics.data.actuator("ur5e/fingers_actuator")

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def get_gripper_closed_value(self):
        """
        Calculates the normalized value representing the closeness of the gripper.
        
        1 means the gripper is fully closed and 0 means the gripper is fully open.

        Returns:
            float: A normalized value between 0 and 1 representing the closeness of the gripper.
        """
        right_driver_joint = self._arm._mjcf_root.find("joint", "right_driver_joint")
        left_driver_joint = self._arm._mjcf_root.find("joint", "left_driver_joint")

        right_driver_pos = self._physics.bind(right_driver_joint).qpos[0]
        left_driver_pos = self._physics.bind(left_driver_joint).qpos[0]

        gripper_closed_value = (right_driver_pos + left_driver_pos) / 2
        gripper_closed_value = np.clip(gripper_closed_value, 0, 0.8) / 0.8

        return gripper_closed_value

    def get_observation(self):
        # Here a collect trajectory script needs a dict with "robot_state" key
        cartesian_position = self._arm.get_eef_pose(self._physics)
        pos = cartesian_position[:3]
        quat = cartesian_position[3:]
        euler = quat2euler(quat)
        cartesian_position_euler = np.concatenate([pos, euler])

        return {
            "timestamp": {},
            "robot_state": {
                "cartesian_position": cartesian_position_euler,
                "gripper_position": self.get_gripper_closed_value(),
            },
        }

    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        # print(dir(self._arm.eef_site))
        return self._arm.get_eef_pose(self._physics)

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def close_grip(self):
        self._griper_actuator.ctrl = 250
        # self._physics.data.actuator("ur5e/fingers_actuator").ctrl = 250

    def open_grip(self):
        self._griper_actuator.ctrl = 0

    def reset(self, seed=None, options=None, randomize=False) -> tuple:
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints[:6]).qpos = [
                0.0,
                -1.5707,
                1.5707,
                -1.5707,
                -1.5707,
                0.0,
                # KC:
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
            ]
            # put target in a reasonable starting position
            self._target.set_mocap_pose(
                self._physics, position=[0.5, 0, 0.1], quaternion=[0, 0, 0, 1]
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        
        # The following code assumes:
        # action[:3] x, y, z target position in cartesian space
        # action[3:6] euler angles target orientation in cartesian space
        # action[6] gripper action 1 means close, 0 means open
        
        target_position = action[:3]
        target_euler = action[3:6]

        target_quat = euler2quat(target_euler)
        target_pose = np.concatenate([target_position, target_quat])

        gripper_action = action[-1]
        if gripper_action > 0.5:
            self.close_grip()
        else:
            self.open_grip()

        # run OSC controller to move to target pose
        self._controller.run(target_pose)

        # step physics
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()

        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()
