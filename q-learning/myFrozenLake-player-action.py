import gym
import numpy as np
import time
import pygame
import sys
from pygame.locals import *


class Game:
    stan = 0;

    def step(self, action):
        reward = -0.04
        done = False
        info = False

        if (action == 0) and ((self.stan % 4) != 0):
            self.stan -= 1
        if (action == 1) and (self.stan < 12):
            self.stan += 4
        if (action == 2) and ((self.stan % 4) != 3):
            self.stan += 1
        if (action == 3) and (self.stan > 3):
            self.stan -= 4

        if (field[self.stan] == 'H'):
            reward = -5
            done = True

        if field[self.stan] == 'G':
            reward = 1
            done = True

        return self.stan, reward, done, info;

    def __init__(self, field):
        self.field = field

    def reset(self):
        self.stan = 0
        return self.stan;


def getKey():
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 0
                if event.key == pygame.K_RIGHT:
                    return 2
                if event.key == pygame.K_UP:
                    return 3
                if event.key == pygame.K_DOWN:
                    return 1


# Grid world init
pygame.init()
font = pygame.font.SysFont("monospace", 29, True)
surface = pygame.display.set_mode((860, 860))  # width x height
pygame.display.set_caption('GridWorld')
sleep_time = 0.01;


def drawGridWorld(Q, field, player, action):
    surface.fill((0, 0, 0))
    wiersz = 0
    kolumna = 0
    offset = 10
    size = 200
    # print(action)
    for pole in range(len(Q)):  # Y # pola pionowo
        if pole != 0 and (pole % len(Q[0]) == 0):
            wiersz += 1
            kolumna = 0
        x_cord = offset + offset * kolumna + kolumna * size
        y_cord = offset + offset * wiersz + wiersz * size
        # Field
        field_color = (189, 189, 189)
        if field[pole] == 'H':
            field_color = (33, 33, 33)
        if field[pole] == 'S':
            field_color = (255, 179, 0)
        if field[pole] == 'G':
            field_color = (118, 255, 3)
        pygame.draw.rect(surface, field_color, (x_cord, y_cord, size, size))
        # Player
        if pole == player:
            field_color = (3, 169, 244)
            pygame.draw.circle(surface, field_color, (int(round(x_cord + size / 2)), int(round(y_cord + size / 2))),
                               int(round(size / 2)))
        if action == 0:
            move_action = font.render("<", False, (255, 0, 0))
        if action == 1:
            move_action = font.render("\/", False, (255, 0, 0))
        if action == 2:
            move_action = font.render(">", False, (255, 0, 0))
        if action == 3:
            move_action = font.render("/\\", False, (255, 0, 0))

        surface.blit(move_action, (0, 0))
        # QMatrix

        color = (255, 255, 255)

        best = Q[pole].argmax()
        for i in range(4):
            # print(best)
            x_label_cord = 0
            y_label_cord = 0
            if i == 0:  # left
                x_label_cord = x_cord
                y_label_cord = y_cord
                direction = 'left'

            if i == 1:  # down
                x_label_cord = x_cord
                y_label_cord = y_cord + size / 4
                direction = 'down'

            if i == 2:  # right
                x_label_cord = x_cord
                y_label_cord = y_cord + size / 4 * 2
                direction = 'right'

            if i == 3:  # up
                x_label_cord = x_cord
                y_label_cord = y_cord + size / 2 + size / 4
                direction = 'up'

            label = font.render("{}:{}".format(direction, round(Q[pole][i], 3)), False, color)
            surface.blit(label, (x_label_cord, y_label_cord))
        kolumna += 1
    pygame.display.update()
    time.sleep(sleep_time)


# env = gym.make('FrozenLake-v0')
field = ['S', 'F', 'F', 'F',
         'F', 'H', 'F', 'H',
         'F', 'F', 'F', 'H',
         'H', 'F', 'F', 'G'
         ]
env = Game(field)
# Initialize table with all zeros
Q = np.zeros([16, 4])

# env.setField(field);
# print(Q.shape)
# print(Q)
# print(len(Q))
# print(len(Q[len(Q)]))
alpha = .8
y = .95
num_episodes = 2000
reward_list = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    observation = env.reset()
    drawGridWorld(Q, field, observation, 0)
    # time.sleep(0.1)
    # print(observation)
    # env.render()
    rewardAll = 0
    done = False
    loop_counter = 0
    # The Q-Table learning algorithm
    while loop_counter < 99:
        loop_counter += 1
        # Choose an action by greedily (with noise) picking from Q table
        # action = np.argmax(Q[observation, :] + np.random.randn(1, 4) * (1. / (i + 1)))
        action = getKey();
        # Get new state and reward from environment
        drawGridWorld(Q, field, observation, action)
        observation1, reward, done, info = env.step(action)
        if (reward == 1) and (sleep_time < 0.01):
            sleep_time += 0.01
        drawGridWorld(Q, field, observation1, action)
        # print(observation1)
        # print(info)
        # Update Q-Table with new knowledge
        Q[observation, action] = Q[observation, action] + alpha * (
            reward + y * np.max(Q[observation1, :]) - Q[observation, action])
        # print(Q)
        # env.render()
        # time.sleep(0.1)
        rewardAll += reward
        observation = observation1

        if done:
            break
    # jList.append(j)
    reward_list.append(rewardAll)
