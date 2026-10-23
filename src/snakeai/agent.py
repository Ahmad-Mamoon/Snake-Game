import os
import random
from collections import deque

import numpy as np
import torch

from .game import BLOCK_SIZE, SPEED, Direction, Point, SnakeGameAI
from .model import MODEL_DIR, Linear_QNet, QTrainer
from .plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
HIDDEN_SIZE = 256
EXPLORE_GAMES = 80


class Agent:

    def __init__(self, lr=LR, gamma=GAMMA, hidden_size=HIDDEN_SIZE,
                 max_memory=MAX_MEMORY, batch_size=BATCH_SIZE,
                 explore_games=EXPLORE_GAMES):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.explore_games = explore_games
        self.memory = deque(maxlen=max_memory)
        self.model = Linear_QNet(11, hidden_size, 3)
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        food = game.food or head

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,

            food.x < head.x,
            food.x > head.x,
            food.y < head.y,
            food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards),
                                np.array(next_states), dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, greedy=False):
        self.epsilon = 0 if greedy else max(0, self.explore_games - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train(max_games=None, show_plot=True, speed=SPEED, render=True, resume=False,
          **agent_kwargs):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(**agent_kwargs)
    if resume and agent.model.load():
        agent.model.train()
        agent.n_games = agent.explore_games  # skip the random-exploration phase
        record = Linear_QNet.saved_record()  # only overwrite the saved model if we beat it
        print(f"Resumed from saved model (record {record}).")
    game = SnakeGameAI(speed=speed, render=render)
    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_path = os.path.join(MODEL_DIR, "training_plot.png")

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(record=record)

            plot_scores.append(score)
            total_score += score
            games_played = len(plot_scores)
            mean_score = total_score / games_played
            plot_mean_scores.append(mean_score)
            print(f"Games: {games_played}  Score: {score}  Best: {record}  "
                  f"Mean: {mean_score:.2f}")
            if show_plot:
                plot(plot_scores, plot_mean_scores, save_path=plot_path)

            if max_games is not None and games_played >= max_games:
                break


if __name__ == "__main__":
    train()
