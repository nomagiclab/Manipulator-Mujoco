import gymnasium
import manipulator_mujoco

# Create the environment with rendering in human mode
env = gymnasium.make("manipulator_mujoco/UR5eEnv-v0", render_mode="human")

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# Run simulation for a fixed number of steps
state = 0
cnt = 0
for _ in range(10000):
    if state == 2 and cnt > 100:
        break
    # action = env.action_space.sample()
    target = [0.5, 0, 0.1 + state * 0.1]
    action = target + [250 if state == 1 else 0]
    observation, reward, terminated, truncated, info = env.step(action)
    x1 = observation[0]
    x2 = observation[1]
    x3 = observation[2]
    err = sum([(x1 - target[0]) ** 2, (x2 - target[1]) ** 2, (x3 - target[2]) ** 2])

    if state == 0 and err < 0.0001:
        state = 1
    elif state == 1 and err < 0.0001:
        state = 2
        cnt = 0

    cnt += 1
    if terminated or truncated:
        observation, info = env.reset()

env.close()
