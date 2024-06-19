# Snake Game AI with Reinforcement Learning

This project implements an AI agent using reinforcement learning to play the classic Snake game. The agent learns to navigate the game environment, collect food, and avoid collisions using a Q-learning approach implemented with PyTorch.

## Project Structure

The project consists of the following Python files:

- **agent.py**: Defines the `Agent` class responsible for the AI agent's behavior, including state handling, action selection, memory management, and training methods.
- **game.py**: Contains the `SnakeGameAI` class which simulates the Snake game environment. It manages game state, user input, collision detection, and scoring.
- **model.py**: Implements the neural network model (`Linear_QNet`) used by the AI agent to predict Q-values. Also defines the `QTrainer` class responsible for training the model.
- **plot.py**: Provides plotting functions using Matplotlib to visualize game scores and mean scores over time during training.
- **snake_game.py**: Is the human playable version of the game.

## Dependencies

- Python 3.x
- Pygame
- NumPy
- PyTorch
- Matplotlib

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/snake-game-ai.git
   cd snake-game-ai

2. **Install dependencies:**
   ```bash
    pip install pygame numpy torch matplotlib

3. **Run the training script:**
    ```bash
    python agent.py

4. **Monitor training:**
   - While the training is running, you can observe the game's progress in the Pygame window.
   - Matplotlib also displays a dynamic plot showing the scores and mean scores over the course of training.

5. **End Training:**
   - You can manually stop training at any point or you can specify the iterations count.
   - The best performing model is saved automatically (model.pth in the ./model folder).

## Customization and Extension
  - **Adjusting Parameters:** You can modify parameters such as MAX_MEMORY, BATCH_SIZE, and LR in train.py and agent.py to optimize training performance.
  - **Enhancing Game Features:** Extend the SnakeGameAI class in game.py to add more game features, different rewards, or more complex game mechanics.
  - **Experimenting with Neural Networks:** Modify the model architecture or try different neural network models in model.py to improve learning and performance.

## Credits
  - The initial structure and training loop of the project were inspired by various tutorials and resources on reinforcement learning and game AI.
