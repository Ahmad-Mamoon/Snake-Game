import random
from collections import deque

import pygame

from .game import (BLACK, BLOCK_SIZE, DIR_VECTORS, RED, WHITE, Direction, Point,
                   font, initial_snake)

COLOR1 = (0, 255, 0)
COLOR2 = (100, 100, 100)
SPEED = 15

KEY_TO_DIR = {
    pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
    pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
    pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
}
OPPOSITE = {
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
}


class SnakeGame:

    def __init__(self, w=640, h=480, speed=SPEED):
        self.w = w
        self.h = h
        self.speed = speed
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = random.choice(list(Direction))
        self.pending = deque(maxlen=3)
        self.snake = initial_snake(self.w, self.h, self.direction)
        self.head = self.snake[0]
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE
        free_cells = [
            Point(c * BLOCK_SIZE, r * BLOCK_SIZE)
            for c in range(cols)
            for r in range(rows)
            if Point(c * BLOCK_SIZE, r * BLOCK_SIZE) not in self.snake
        ]
        self.food = random.choice(free_cells) if free_cells else None

    def queue_direction(self, new_dir):
        last = self.pending[-1] if self.pending else self.direction
        if new_dir != last and new_dir != OPPOSITE[last]:
            self.pending.append(new_dir)

    def play_step(self):
        """Advance one frame. Returns (game_over, score)."""
        if self.pending:
            self.direction = self.pending.popleft()

        dx, dy = DIR_VECTORS[self.direction]
        self.head = Point(self.head.x + dx * BLOCK_SIZE, self.head.y + dy * BLOCK_SIZE)
        self.snake.insert(0, self.head)

        if self._is_collision():
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
            if self.food is None:
                return True, self.score
        else:
            self.snake.pop()
        return False, self.score

    def _is_collision(self):
        if (self.head.x > self.w - BLOCK_SIZE or self.head.x < 0
                or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0):
            return True
        return self.head in self.snake[1:]

    def draw(self, message=None):
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
        self.display.blit(font.render(f"Score: {self.score}", True, WHITE), [0, 0])
        if message:
            text = font.render(message, True, WHITE)
            self.display.blit(text, text.get_rect(center=(self.w // 2, self.h // 2)))
        pygame.display.flip()


def main(speed=SPEED, w=640, h=480):
    game = SnakeGame(w=w, h=h, speed=speed)
    best = 0
    paused = False
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                print(f"Best score this session: {best}")
                return
            if event.type != pygame.KEYDOWN:
                continue
            if game_over and event.key in (pygame.K_r, pygame.K_SPACE):
                game.reset()
                game_over = False
            elif event.key in (pygame.K_p, pygame.K_SPACE):
                paused = not paused
            elif event.key in KEY_TO_DIR and not paused:
                game.queue_direction(KEY_TO_DIR[event.key])

        if game_over:
            game.draw("Game over! R to restart, Esc to quit")
        elif paused:
            game.draw("Paused - P to resume")
        else:
            game_over, score = game.play_step()
            if game_over:
                best = max(best, score)
                print(f"Game over. Score: {score}  Best: {best}")
            game.draw()
        game.clock.tick(game.speed)


if __name__ == "__main__":
    main()
