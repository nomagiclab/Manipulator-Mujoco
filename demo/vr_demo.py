from manipulator_mujoco.envs import UR5eEnv
from droid.controllers.oculus_controller import VRPolicy
from droid.trajectory_utils.misc import collect_trajectory

class StubPolicy:
    def forward(self, obs, include_info=False):
        return [0.01, 0.01, 0.01, 0.01, 0.01, 0.01], {}
    def reset_state(self):
        pass
    def get_info(self):
        return {"movement_enabled": True}

# Make the robot env
env = UR5eEnv(render_mode="human")
controller = VRPolicy()
# controller = StubPolicy()

# print("Ready")
collect_trajectory(env, controller=controller)