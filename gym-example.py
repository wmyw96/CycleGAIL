import gym
import time

env = gym.make('Pendulum-v0')
env.reset()

print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)

for i in range(100):
    env.render()
    act = env.action_space.sample()
    print(act)
    env.step(act)
    time.sleep(1.0)
