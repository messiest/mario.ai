import gym 
env = gym.make('SuperMarioBros-1-2-v0')
observation = env.reset()
for i in range(10000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print(i, info)
quit()
