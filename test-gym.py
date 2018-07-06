import gym
env = gym.make('Ant-v1')

#for i_episode in range(20):
#    observation = env.reset()
#    print(env.)
#    for t in range(100):
#        env.render()
#        print(observation)
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(action)
#        print('reward in %d timesteps: %.5f\n' % (t + 1, reward))
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
