# Deep Reinforcement Learning with Pong Pygame

Welcome to my Deep Reinforcement Learning Pong Project. 
The goal of this project is to capture the journey of how an AI agent goes from not knowing what Pong is to being UNDEFEATABLE :)

## The Development Journey
From the very beginning, I didn't know exactly how long it takes for the agent to learn Pong. So I did a lot of research and decided to make this a 4-day project, and I trained the model overnight every day and it went through over 12,000 games against a hard-coded computer. And the final challenge is to play against a human (me). 

### Day 1: The Newbie
On day one, the focus was on setting up the environment and getting a taste of building a deep learning model. Using Pygame, the classic Pong game was simulated. The paddles were set up, the ball was bouncing around, and the game mechanics were in place. I decided to use a policy network with a policy gradient approach. The reason is that the policy gradient model allows the agent to learn directly and continuously from its interactions with the environment without requiring a predefined strategy. 

It works! Although it seems very clueless, I was excited to see what happens tomorrow after a night of training.

<img src=![Day1_PDRL](https://github.com/user-attachments/assets/059e1bcc-8242-4b7c-86d1-77637b5e1a11) alt="Description" width="300" height="200">
![Day1_DPRL_Clueless_GIF](https://github.com/user-attachments/assets/bb988df9-1636-4375-9bf3-c80d07012ec6)

### Day 2: The Awakening (The Ball Matters!)
Surprisingly when I woke up, it went through 7,500 games, and the agent learned how to HIT the ball! It became clear that the most crucial piece of information for the agent was the ball. The agent needed to focus on the ball's movement, predict where it would go, and react accordingly. With only two layers, the network began learning from the game state: paddle positions, ball coordinates, and velocities. I was very happy to see that the agent finally started moving the paddle, though it wasn't winning any games yet.

[photo of 7500 games] [video of day2 game]

### Day 3: Learning to Win (Victory Awaits)
Day three was all about refining the agent's ability to win. I found that the reward system wasn't too balanced, so I lowered the reward of hitting the ball from 0.5 to 0.1 and kept the 1.0 reward for scoring a point. The agent now understood the concept of reward: hitting the ball is good, but gaining a big reward for scoring points is the ultimate goal. While still, I kept some negative rewards for missing the ball and conceding a point.

After many episodes of trial and error, the agent began to improve. It moved more purposefully and started making strategic decisions to maximize its chances of winning. It started to be competitive against the hardcoded computer! 

[photo of reward system] [video of day3 game]

### Day 4: Testing the AI's Ability
On day four, I woke up seeing "1492 : 1678" - the agent was winning against the computer. Now, it's time to test. AI's ability was put to the test in a real game against humans (me)...
A ten-point match was played against the agent, and the result was an..., 8:10 loss. The agent demonstrated strong gameplay, consistently predicting ball movements and outplaying me in several key moments.

Let's watch the video - 
[video of the final game]

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
- **Rewards:** The agent receives positive rewards for hitting the ball (0.5 -> 0.1) and scoring points (1.0), and negative rewards for letting the opponent hit the ball (-0.5 -> -0.1) and scoring points (-1.0).

## Potential Improvements
Although the agent is performing well, there are still several areas that could be improved:

1. **More Complex Neural Network:** The current policy network is relatively simple. Adding more layers or using convolutional layers could help the agent extract more complex features from the game state.
2. **Value Function Approximation:** Implementing an Actor-Critic approach could help the agent stabilize training and learn more efficiently by having a separate value function that predicts expected rewards.
3. **Exploration Strategies:** The current model selects actions based on the output probabilities from the policy network. Implementing an epsilon-greedy strategy or using entropy regularization could encourage more exploration, helping the agent discover better strategies.
4. **Environment Augmentation:** Currently, the environment is quite static. Adding random elements, such as varying ball speed, could make the agent more robust and better at generalizing its strategies.

Feel free to explore the project further and contribute if you have ideas for improvement!

## How to Run the Project
To run the project, ensure that you have Python, Pygame, PyTorch, and other necessary libraries installed. Simply execute the main script, and watch as the RL agent learns to play Pong!

```bash
python main.py
```

Go ahead and enjoy watching the AI learn and improve, thanks for reading!

