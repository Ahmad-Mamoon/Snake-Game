import os
import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")
FONT_PATH = os.path.join(ASSETS_DIR, "arial.ttf")

pygame.init()
if os.path.exists(FONT_PATH):
    font = pygame.font.Font(FONT_PATH, 25)
else:
    font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

WHITE = (255, 255, 255)
RED = (200, 0, 0)
COLOR1 = (0, 0, 255)
COLOR2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 50

DIR_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
}


def initial_snake(w, h, direction):
    """Head at the board centre with the body trailing opposite to `direction`."""
    cx = (w // BLOCK_SIZE // 2) * BLOCK_SIZE
    cy = (h // BLOCK_SIZE // 2) * BLOCK_SIZE
    dx, dy = DIR_VECTORS[direction]
    return [Point(cx - i * dx * BLOCK_SIZE, cy - i * dy * BLOCK_SIZE) for i in range(3)]


class SnakeGameAI:

    def __init__(self, w=640, h=480, speed=SPEED, render=True):
        self.w = w
        self.h = h
        self.speed = speed
        self.render = render
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = random.choice(list(Direction))
        self.snake = initial_snake(self.w, self.h, self.direction)
        self.head = self.snake[0]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE
        free_cells = [
            Point(c * BLOCK_SIZE, r * BLOCK_SIZE)
            for c in range(cols)
            for r in range(rows)
            if Point(c * BLOCK_SIZE, r * BLOCK_SIZE) not in self.snake
        ]
        if not free_cells:
            self.food = None
            return
        self.food = random.choice(free_cells)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            if self.food is None:  # board is full: the snake has won
                return reward, True, self.score
        else:
            self.snake.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if (pt.x > self.w - BLOCK_SIZE or pt.x < 0
                or pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, COLOR1,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, COLOR2,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        if self.food is not None:
            pygame.draw.circle(
                self.display, RED,
                (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2),
                BLOCK_SIZE // 2,
            )
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
