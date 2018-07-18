state = env.reset()
init_qpos = np.reshape(env.env.model.data.qpos, (-1))
init_vel = np.reshape(env.env.model.data.qvel, (-1))
actions = []
states = []
reward_sum = 0
for t in range(10000): # Don't infinite loop while learning
    action = select_action(state)
    action = action.data[0].numpy()
    actions.append(action)
    states.append(state)
    next_state, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print(t)
        break
    state = next_state
print(reward_sum)


reward_sum = 0
err = 0.0
def set_state(env, state):
    qpos = np.concatenate([np.zeros(1) * 2.34, state[:8]])
    qv = state[8:]
    env.env.set_state(qpos, qv)
env.reset()
env.env.set_state(init_qpos, init_vel)
state = states[0]
for i in range(10000):
    err += np.sum((states[i] - state) * (states[i] - state))
    state, rd, done, _ = env.step(actions[i])
    reward_sum += rd
    if done:
        print(i)
        break
print(reward_sum)
print(err)