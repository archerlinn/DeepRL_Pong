import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import math
from torch.utils.tensorboard import SummaryWriter


# ==============================
# 1. Initialize Pygame and Define Constants
# ==============================

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong RL")

# Define Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define game objects dimensions
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 20

# Initialize paddles and ball
paddle1 = pygame.Rect(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)  # Left Paddle (Hardcoded AI)
paddle2 = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)  # Right Paddle (RL Agent)
ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)

# Ball velocity
ball_velocity = [10, 10]

# Define actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_STAY = 2
NUM_ACTIONS = 3

# Frame rate
clock = pygame.time.Clock()
FPS = 60  # Frames per second

# Scores
score1 = 0  # Left Paddle (AI)
score2 = 0  # Right Paddle (RL Agent)

# Font for displaying score
font = pygame.font.Font(None, 74)

# ==============================
# 2. Define Game Mechanics
# ==============================

def reset_ball():
    """Reset the ball to the center with a random direction."""
    global ball_velocity
    ball.center = (WIDTH // 2, HEIGHT // 2)
    angle = random.uniform(0, 2 * math.pi)
    initial_speed = 9
    ball_velocity = [initial_speed * math.cos(angle), initial_speed * math.sin(angle)]
    # Ensure the ball isn't moving too vertically or horizontally
    while abs(ball_velocity[0]) < 3 or abs(ball_velocity[1]) < 3:
        angle = random.uniform(0, 2 * math.pi)
        ball_velocity = [initial_speed * math.cos(angle), initial_speed * math.sin(angle)]

def reset_game():
    """Reset the game to the initial state."""
    global paddle1, paddle2, ball, ball_velocity, score1, score2
    paddle1.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    paddle2.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    reset_ball()
    score1 = 0
    score2 = 0
    return get_state()

def get_state():
    """
    Get the current state of the game.
    State consists of normalized positions and velocities:
    [paddle1_y, paddle2_y, ball_x, ball_y, ball_vel_x, ball_vel_y]
    """
    state = np.array([
        paddle1.y / HEIGHT,
        paddle2.y / HEIGHT,
        ball.x / WIDTH,
        ball.y / HEIGHT,
        ball_velocity[0] / 10,  # Assuming max velocity ~10
        ball_velocity[1] / 10
    ], dtype=np.float32)
    return state

def step_game(action, render=False):
    """
    Perform a game step with the given action for paddle2 (RL Agent).
    
    Args:
        action (int): The action taken by the RL agent.
        render (bool): Whether to render the game step.
    
    Returns:
        tuple: (next_state, reward, done)
    """
    global paddle1, paddle2, ball, ball_velocity, score1, score2

    # Define paddle movement speed
    paddle_speed = 7

    # RL Agent's action (Right Paddle)
    if action == ACTION_UP:
        paddle2.y -= paddle_speed
    elif action == ACTION_DOWN:
        paddle2.y += paddle_speed
    # ACTION_STAY does nothing

    # Ensure paddle2 stays on screen
    paddle2.y = max(0, min(HEIGHT - PADDLE_HEIGHT, paddle2.y))

    # Hardcoded AI for paddle1 (Left Paddle): Move paddle towards the ball
    opponent_speed = 7
    if paddle1.centery < ball.centery and paddle1.bottom < HEIGHT:
        paddle1.y += opponent_speed
    elif paddle1.centery > ball.centery and paddle1.top > 0:
        paddle1.y -= opponent_speed

    # Update ball position
    ball.x += ball_velocity[0]
    ball.y += ball_velocity[1]

    # Collision with top and bottom walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_velocity[1] = -ball_velocity[1]

    # Initialize reward
    reward = 0
    done = False

    # Collision with paddles
    if ball.colliderect(paddle1):
        ball_velocity[0] = abs(ball_velocity[0])  # Ensure ball moves right
        # Slightly adjust the vertical velocity based on paddle position
        delta = (ball.centery - paddle1.centery) / (PADDLE_HEIGHT / 2)
        ball_velocity[1] += delta
        # Intermediate Punishment for hitting the ball
        reward -= 0.5
        print(f"[DEBUG] Opponent Hits the Ball. Reward: {reward:.2f}")
    elif ball.colliderect(paddle2):
        ball_velocity[0] = -abs(ball_velocity[0])  # Ensure ball moves left
        delta = (ball.centery - paddle2.centery) / (PADDLE_HEIGHT / 2)
        ball_velocity[1] += delta
        # Intermediate Reward for hitting the ball
        reward += 0.5
        print(f"[DEBUG] Agent Hit the Ball. Reward: {reward:.2f}")

    # Limit the vertical speed to prevent it from becoming too fast
    max_ball_speed = 15
    ball_velocity[1] = max(-max_ball_speed, min(max_ball_speed, ball_velocity[1]))

    # Check for scoring
    if ball.left <= 0:
        score1 += 1
        reward += 1  # Positive reward for RL agent by scoring a point
        done = True
        print(f"[DEBUG] Agent Scored a Point. Reward: {reward:.2f}")
    elif ball.right >= WIDTH:
        score2 += 1
        reward -= 1   # Negative reward for RL agent by conceding a point
        done = True
        print(f"[DEBUG] Agent Conceded a Point. Reward: {reward:.2f}")

    # Optionally render the game
    if render:
        render_game()

    next_state = get_state()
    return next_state, reward, done

def render_game():
    """Render the current game state to the Pygame window."""
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, paddle1)
    pygame.draw.rect(screen, WHITE, paddle2)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    # Display scores
    score_text1 = font.render(str(score1), True, WHITE)
    score_text2 = font.render(str(score2), True, WHITE)
    screen.blit(score_text1, (WIDTH // 2 - 50, 10))
    screen.blit(score_text2, (WIDTH // 2 + 20, 10))

    pygame.display.flip()
    clock.tick(FPS)

# ==============================
# 3. Define the Policy Network and Agent
# ==============================

class PolicyNetwork(nn.Module):
    """
    A simple neural network with one hidden layer for the Policy Gradient agent.
    """
    def __init__(self, input_size=6, hidden_size=128, output_size=3):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_probs = F.softmax(x, dim=1)
        return action_probs

class PolicyGradientAgent:
    def __init__(self, lr=1e-3, gamma=0.99, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Debugging line
        
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        
        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir='runs/PongRL')
                
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Move state to GPU
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()
    
    def store_reward(self, reward):
        """
        Store the reward received after taking an action.
        
        Args:
            reward (float): The reward received.
        """
        self.rewards.append(reward)
    
    def reset(self):
        """Reset the stored log probabilities and rewards."""
        self.log_probs = []
        self.rewards = []
    
    def compute_returns(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns
    
    def update_policy(self, episode):
        """
        Update the policy network using the collected log probabilities and rewards.
        
        Args:
            episode (int): The current episode number (for logging purposes).
        """
        returns = self.compute_returns()
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Log the policy loss and total reward to TensorBoard
        self.writer.add_scalar('Loss/Policy Loss', policy_loss.item(), episode)
        self.writer.add_scalar('Reward/Total Reward', sum(self.rewards), episode)
        
        self.reset()
    
    def close_writer(self):
        """Close the TensorBoard writer."""
        self.writer.close()


# ==============================
# 4. Main Training Loop
# ==============================

def main():
    """
    Main function to run the training loop.
    """
    agent = PolicyGradientAgent()

    try:
        agent.policy_net.load_state_dict(torch.load("policy_net_saved_episode.pth"))
        print("Loaded pre-trained model weights.")
    except FileNotFoundError:
        print("No pre-trained model found. Starting training from scratch.")

    num_episodes = 10000000
    max_steps = 10000  # Maximum steps per episode changed from 1000 to 10000

    # Initial game reset
    state = reset_game()

    for episode in range(1, num_episodes + 1):
        state = reset_game()
        total_reward = 0
        done = False
        step_num = 0

        while not done and step_num < max_steps:
            # Select and perform an action
            action = agent.select_action(state)
            # Render the game for visualization
            render = True  # Set to True to always render
            next_state, reward, done = step_game(action, render=render)
            agent.store_reward(reward)
            total_reward += reward
            state = next_state
            step_num += 1

            # Handle Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        # Pause the game and wait until unpaused
                        paused = True
                        pause_menu()

        # Update the policy after each episode
        if episode % 20 == 0:
            agent.update_policy(episode)
            print("----------------!policy updated!----------------")
        print(f"Episode {episode}\tTotal Reward: {total_reward:.2f}\tScore: {score1}-{score2}")
        
        # Save the policy network every 100 episodes
        if episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"policy_net_saved_episode.pth")

    pygame.quit()

def pause_menu():
    """
    Displays a pause menu allowing the user to resume or quit the game.
    """
    paused = True
    menu_font = pygame.font.Font(None, 50)
    while paused:
        screen.fill(BLACK)
        pause_text = menu_font.render("Game Paused. Press P to Resume or Q to Quit.", True, WHITE)
        screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - pause_text.get_height() // 2))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        
        clock.tick(FPS)

if __name__ == "__main__":
    main()
