from manipulator_mujoco.envs import UR5eEnv
from droid.controllers.oculus_controller import VRPolicy
from droid.trajectory_utils.misc import collect_trajectory

# Make the robot env
env = UR5eEnv(render_mode="human")
controller = VRPolicy()

# print("Ready")
collect_trajectory(env, controller=controller)