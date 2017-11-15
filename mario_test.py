import gym 
env = gym.make('SuperMarioBros-1-1-v0')
observation = env.reset()
for _ in range(10000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
