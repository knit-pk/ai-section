import gym
import numpy as np
import time


def example1():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):  # tyle będzie razy uruchomiona gra
        observation = env.reset()
        for t in range(200):  # tyle razy w każdej grze będzie
            env.render()  # podgląd
            print(observation)  # wypisz obserwacje
            # Num	    Observation	            Min	    Max
            # 0	        Cart Position	        -2.4	2.4
            # 1	        Cart Velocity	        -Inf	Inf
            # 2	        Pole Angle	            ~-41.8°	~41.8°
            # 3	        Pole Velocity At Tip	-Inf	Inf
            action = env.action_space.sample()
            # Num	Action
            # 0	    Push cart to the left
            # 1	    Push cart to the right
            observation, reward, done, info = env.step(action)  # podejmowana jest losowa akcja
            if done:
                print("Episode finished after {} timesteps, Reward {}".format(t + 1, reward))
                break


def example2(delay):
    env = gym.make('CartPole-v0')
    for i_episode in range(20):  # tyle będzie razy uruchomiona gra
        observation = env.reset()
        for t in range(200):  # tyle razy w każdej grze będzie
            time.sleep(delay)
            env.render()  # podgląd
            # print(observation)  # wypisz obserwacje
            # Num	    Observation	            Min	    Max
            # 0	        Cart Position	        -2.4	2.4
            # 1	        Cart Velocity	        -Inf	Inf
            # 2	        Pole Angle	            ~-41.8°	~41.8°
            # 3	        Pole Velocity At Tip	-Inf	Inf
            action = env.action_space.sample()
            # Num	Action
            # 0	    Push cart to the left
            # 1	    Push cart to the right
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps, Reward {}".format(t + 1, reward))
                break


def run_episode(env, parameters):
    observation = env.reset()
    # env.render()
    totalreward = 0
    for _ in range(200):
        # env.render()  # podgląd
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        # print(observation)
        totalreward += reward
        if done:
            break
    return totalreward


def random_search():
    env = gym.make('CartPole-v0')
    bestparams = env.reset()
    bestreward = 0
    for _ in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        print(reward)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                print("Best reward: {}".format(bestreward))
                print("Parameters {}".format(bestparams))
                break
    return bestparams



def play():
    env = gym.make('CartPole-v0')
    # parameters = [0.15985413, 0.81339801, 0.38618734, 0.93636902] #random
    # parameters = [0.82927842, -0.72575277, 0.506987, 0.4658739]  # hill climbing
    # parameters = [ 0.18313874 , 0.18419612 , -0.64222001, 0.99853596] # hill climbing
    parameters = [0.37773476, 0.49696023, 0.70566396, 0.83321323]
    observation = env.reset()
    bestreward = 0
    for _ in range(10000):
        time.sleep(0.03)
        env.render()  # podgląd
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        bestreward = bestreward + reward
        if done:
            print("Reward: {}".format(bestreward))
            break


def play_with_parameters(parameters, how_many, delay):
    env = gym.make('CartPole-v0')
    for i in range(how_many):
        observation = env.reset()
        bestreward = 0
        for _ in range(10000):
            time.sleep(delay)
            if delay > 0:
                env.render()  # podgląd
            action = 0 if np.matmul(parameters, observation) < 0 else 1
            observation, reward, done, info = env.step(action)
            bestreward = bestreward + reward
            if done:
                print("Reward: {}".format(bestreward))
                break
    return bestreward


def hill_climbing(noise_scaling):
    env = gym.make('CartPole-v0')
    parameters = np.random.rand(4) * 2 - 1
    bestreward = 0
    reward = 0
    for _ in range(10000):
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = run_episode(env, newparams)
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == 200:
                print(parameters)
                break
    print("Best reward {}, best params {}".format(bestreward, parameters))
    return parameters


if __name__ == '__main__':
    # example1()
    # example2(0.03)

    # random_search()

    # play_with_parameters(random_search(), 1, 0.03)  # 0.03 ~= 30FPS

    # parameters = hill_climbing(0.1)

    # play_with_parameters(hill_climbing(0.1), 1, 0.03)

    # for j in range(10):
    #     parameters = hill_climbing(0.1)
    #     print('Param {}, iteration {}'.format(parameters, j))
    #     play_with_parameters(parameters, 4, 0.03)
