import pickle
import gymnasium
import manipulator_mujoco
from PIL import Image

from manipulator_mujoco.action_modes.absolute_end_effector_pose import (
    AbsoluteEndEffectorPose,
)
from tasks.reach_target import ReachTarget

# Create the environment with rendering in human mode
env = gymnasium.make(
    "manipulator_mujoco/UR5eEnv-task",
    action_mode=AbsoluteEndEffectorPose(),
    human_control=False,
    task_class=ReachTarget,
    render_mode="human",
)

observation, info = env.reset(seed=42)

with open("demo.pkl", "rb") as f:
    demo = pickle.load(f)

for i, observation in enumerate(demo[1:]):
    env.step(demo[i - 1]["eef_pose"])

env.close()
