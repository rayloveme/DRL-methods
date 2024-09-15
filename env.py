import gymnasium as gym
from ppo import ClipPPO

scenarios = ['Pendulum-v1']
env= gym.make(scenarios[0])

ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]

num_episodes = 1000
num_steps = 200
batch_size = 64
# no parallel

agent=ClipPPO(ob_dim, ac_dim)

best_reward = -1000
for episode_i in range(num_episodes):
    ob, info = env.reset()
    done = False
    episode_reward = 0

    for step_i in range(num_steps):
        action, value = agent.get_action(ob)
        ob_next, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        truncated = True if step_i == num_steps - 1 else False
        done = terminated or truncated
        agent.store(ob, action, reward, value, done)

        if done:
            break
        ob = ob_next

    agent.update()
    if  episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy(f'./ppo_policy_{scenarios[0]}.pth')
    print(f'Episode {episode_i}, Reward {episode_reward}, Best Reward {best_reward}')






