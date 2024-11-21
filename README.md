# Deep Reinforcement Learning with Pong Pygame
# Pong Deep Reinforcement Learning Project

Welcome to the Pong Deep Reinforcement Learning (RL) Project! This project captures the journey of creating an AI agent that can play Pong, learning from the basics to mastering the game.

## The Development Journey

### Training Overview
The agent was trained overnight every day from Day 1 to Day 3, allowing it to learn from a large number of game episodes and progressively improve its performance.

### Day 1: The Newbie
On day one, the focus was on setting up the environment. Using Pygame, the classic Pong game was simulated. The paddles were set up, the ball was bouncing around, and the game mechanics were in place.

The initial goal was simple: get a feel for the game environment. A framework for controlling the right paddle was established, ensuring that all game objects could interact smoothly. At this stage, the agent was clueless but ready to learn.

### Day 2: The Awakening (The Ball Matters!)
On day two, it became clear that the most crucial piece of information for the agent was the ball. The agent needed to focus on the ball's movement, predict where it would go, and react accordingly. The agent could observe the position of the paddles, the ball, and their respective velocities.

The first version of the policy network was built with a simple feedforward neural network. With only two layers, the network began learning from the game state: paddle positions, ball coordinates, and velocities. The agent finally started moving the paddle, though it wasn't winning any games yet.

### Day 3: Learning to Win (Victory Awaits)
Day three was all about refining the agent's ability to win. A policy gradient approach using the REINFORCE algorithm was implemented, allowing the agent to learn from each action's outcome. The agent now understood the concept of reward: gaining a positive reward for hitting the ball or scoring points, and a negative reward for missing the ball.

After many episodes of trial and error, the agent began to improve. It moved more purposefully and started making strategic decisions to maximize its chances of winning. The agent eventually achieved its first victory, showcasing significant progress.

### Day 4: Testing the AI's Ability
On day four, the AI's ability was put to the test in a real game. A ten-point match was played against the agent, and the result was an 8:10 loss. The agent demonstrated strong gameplay, consistently predicting ball movements and outplaying the opponent in several key moments.

Below are some videos and snapshots from the agent's journey, showcasing its progression from random paddle movements to mastering the art of Pong.

## The Model and Network Architecture
The reinforcement learning agent uses a simple policy network, defined as follows:

### Policy Network Architecture
- **Input Layer:** 6 neurons, representing the game state: paddle1 position, paddle2 position, ball x and y positions, and ball x and y velocities.
- **Hidden Layer:** 128 neurons, fully connected with ReLU activation function.
- **Output Layer:** 3 neurons, representing the action probabilities for moving up, moving down, or staying still.

The network takes the game state as input and outputs a probability distribution over the three possible actions, which the agent samples to decide its next move. The training uses a policy gradient method with a discount factor (gamma) of 0.99 to ensure that future rewards are also taken into account.

### Training Process
- **Loss Function:** The agent's loss function is calculated using the negative log probability of the action taken, weighted by the expected return (cumulative future reward).
- **Optimization:** The Adam optimizer with a learning rate of 1e-3 is used to update the network weights, helping the agent learn effective strategies over time.
- **Rewards:** The agent receives positive rewards for hitting the ball and scoring points, and negative rewards for missing the ball or letting the opponent score.

## Potential Improvements
Although the agent is performing well, there are still several areas that could be improved:

1. **More Complex Neural Network:** The current policy network is relatively simple. Adding more layers or using convolutional layers could help the agent extract more complex features from the game state.
2. **Value Function Approximation:** Implementing an Actor-Critic approach could help the agent stabilize training and learn more efficiently by having a separate value function that predicts expected rewards.
3. **Exploration Strategies:** The current model selects actions based on the output probabilities from the policy network. Implementing an epsilon-greedy strategy or using entropy regularization could encourage more exploration, helping the agent discover better strategies.
4. **Environment Augmentation:** Currently, the environment is quite static. Adding random elements, such as varying paddle speed or random obstacles, could make the agent more robust and better at generalizing its strategies.

Feel free to explore the project further and contribute if you have ideas for improvement!

## How to Run the Project
To run the project, ensure that you have Python, Pygame, PyTorch, and other necessary libraries installed. Simply execute the main script, and watch as the RL agent learns to play Pong!

```bash
python pong_rl.py
```

Enjoy watching the AI learn and improve!


