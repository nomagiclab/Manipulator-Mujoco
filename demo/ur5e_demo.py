import gymnasium
import manipulator_mujoco  # noqa

# Create the environment with rendering in human mode
env = gymnasium.make("manipulator_mujoco/UR5eEnv-v0", render_mode="rgb_array")

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

state = 0
cnt = 0
box_target = info["box_pos"]
state_diff = [0, -0.08, -0.08, 0]
closing = [False, False, True, True]
for _ in range(10000):
    if cnt > 100:
        cnt = 0
        state += 1
        if state >= len(state_diff):
            break

    target = box_target.copy()
    target[2] += state_diff[state]
    action = target + [250 if closing[state] else 0]
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, action, info)
    env.render()
    x1 = observation[0]
    x2 = observation[1]
    x3 = observation[2]
    err = sum([(x1 - target[0]) ** 2, (x2 - target[1]) ** 2, (x3 - target[2]) ** 2])

    if err < 0.0001:
        cnt += 1
    if terminated or truncated:
        observation, info = env.reset()

env.close()
