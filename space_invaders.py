import gym
env = gym.make('SpaceInvaders-v0')

for i in range(100):
    observation = env.reset()
    for i in range(10000):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print(i, info)
        if info['ale.lives'] == 0:
            break

