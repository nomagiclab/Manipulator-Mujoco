import pickle
import gymnasium
import manipulator_mujoco
from PIL import Image

from manipulator_mujoco.action_modes.delta_end_effector_pose import DeltaEndEffectorPose
from tasks.reach_target import ReachTarget

# Create the environment with rendering in human mode
env = gymnasium.make(
    "manipulator_mujoco/UR5eEnv-task",
    action_mode=DeltaEndEffectorPose(),
    human_control=False,
    task_class=ReachTarget,
    render_mode="human",
)

observation, info = env.reset(seed=42)

demo = env.get_demo()

with open("demo.pkl", "wb") as f:
    pickle.dump(demo, f)

env.close()
