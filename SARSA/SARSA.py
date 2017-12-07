'''
Example implementation of SARSA algorithm for learning the path through frozen lake.
The is_slippery flag lets us change the rules of the game, if True the probability of
changing the chosen direction is 4 out of 10.
'''

import gym
import numpy as np
import time
import pygame


class Game:
    stan = 0;

    def __init__(self, field):
        self.field = field

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

        if self.field[self.stan] == 'H':
            reward = -5
            done = True

        if self.field[self.stan] == 'G':
            reward = 1
            done = True

        return self.stan, reward, done, info;

    def reset(self):
        self.stan = 0
        return self.stan;


def drawGridWorld(Q, field, player, action):
    # Grid world init
    pygame.init()
    font = pygame.font.SysFont("monospace", 30, True)
    surface = pygame.display.set_mode((860, 860))  # width x height
    pygame.display.set_caption('GridWorld')
    sleep_time = 0.02;

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
            pygame.draw.circle(surface, field_color, (
                int(round(x_cord + size / 2)), int(round(y_cord + size / 2))),
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
            if i == best:
                color = (255, 0, 0)
            x_label_cord = 0
            y_label_cord = 0
            if i == 0:  # left
                x_label_cord = x_cord
                y_label_cord = y_cord
                direction = 'left'
                # color = (0, 0, 255)  # blue

            if i == 1:  # down
                x_label_cord = x_cord
                y_label_cord = y_cord + size / 4
                direction = 'down'
                # color = (0, 255, 0)  # green

            if i == 2:  # right
                x_label_cord = x_cord
                y_label_cord = y_cord + size / 4 * 2
                direction = 'right'
                # color = (0, 255, 255)  # green blue

            if i == 3:  # up
                x_label_cord = x_cord
                y_label_cord = y_cord + size / 2 + size / 4
                direction = 'up'
                # color = (255, 0, 0)  # red

            label = font.render("{}:{}".format(direction, round(Q[pole][i], 3)), False,
                                color)
            surface.blit(label, (x_label_cord, y_label_cord))
        kolumna += 1
    pygame.display.update()
    time.sleep(sleep_time)


def learn(is_slippery):
    if is_slippery:
        env = gym.make('FrozenLake-v0')
        Q = np.zeros([env.observation_space.n, env.action_space.n])
    else:
        field = ['S', 'F', 'F', 'F',
                 'F', 'H', 'F', 'H',
                 'F', 'F', 'F', 'H',
                 'H', 'F', 'F', 'G'
                 ]
        env = Game(field)
        Q = np.zeros([16, 4])

    a = .8  # alpha
    y = .95  # gamma
    num_episodes = 2000

    for i in range(num_episodes):

        current_state = env.reset()
        current_action = np.argmax(Q[current_state, :])
        for j in range(100):

            next_state, reward, done, _ = env.step(current_action)

            if is_slippery:
                next_action = np.argmax(
                    Q[next_state, :] + np.random.randn(1, env.action_space.n) * (
                        1. / (i + 1)))
            else:
                next_action = np.argmax(Q[next_state, :] + np.random.randn(1, 4) * (
                    1. / (i + 1)))

            Q[current_state, current_action] += a * (
                reward + y * Q[next_state, next_action] - Q[
                    current_state, current_action])

            current_state = next_state
            current_action = next_action

            if done == True:
                break

    return Q


def play(inQ, is_slippery):
    field = ['S', 'F', 'F', 'F',
             'F', 'H', 'F', 'H',
             'F', 'F', 'F', 'H',
             'H', 'F', 'F', 'G'
             ]

    if is_slippery:
        env = gym.make('FrozenLake-v0')
    else:
        env = Game(field)

    num_episodes = 2000
    Q = inQ
    rList = []  # reward list

    for i in range(num_episodes):
        total_reward = 0

        state = env.reset()

        drawGridWorld(Q, field, state, 0)

        action = np.argmax(Q[state, :])
        for j in range(100):

            drawGridWorld(Q, field, state, action)

            state, reward, done, _ = env.step(action)

            action = np.argmax(Q[state, :])

            total_reward += reward

            if done == True:
                break
        rList.append(total_reward)

    print("Score over time: " + str(sum(rList) / num_episodes))


if __name__ == '__main__':
    is_slippery = False
    Q = learn(is_slippery)
    play(Q, is_slippery)
