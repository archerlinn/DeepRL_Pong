import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Define game objects
paddle1 = pygame.Rect(50, 250, 10, 100)  # Left paddle (AI or Player 1)
paddle2 = pygame.Rect(740, 250, 10, 100)  # Right paddle (Player 2)
ball = pygame.Rect(WIDTH // 2 - 10, HEIGHT // 2 - 10, 20, 20)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Paddle speed
paddle_speed = 7
ai_speed = 6

# Ball speed settings
max_speed = 15
initial_speed = 7

# Scores
score1 = 0
score2 = 0

# Font for displaying score and menu
font = pygame.font.Font(None, 74)
menu_font = pygame.font.Font(None, 50)

# Game state
game_type = "Player vs AI"  # Default game type
paused = False

# Function to reset the ball with a random direction
def reset_ball():
    global ball_speed_x, ball_speed_y
    ball.x, ball.y = WIDTH // 2 - 10, HEIGHT // 2 - 10
    angle = random.uniform(0, 2 * math.pi)
    ball_speed_x = initial_speed * math.cos(angle)
    ball_speed_y = initial_speed * math.sin(angle)
    while abs(ball_speed_x) < 3 or abs(ball_speed_y) < 3:
        angle = random.uniform(0, 2 * math.pi)
        ball_speed_x = initial_speed * math.cos(angle)
        ball_speed_y = initial_speed * math.sin(angle)

# Function to display the menu and select game type
def show_menu():
    global game_type, paused
    menu_running = True
    while menu_running:
        screen.fill(BLACK)
        title_text = menu_font.render("Choose Game Type:", True, WHITE)
        option1_text = menu_font.render("1. Player vs AI", True, WHITE)
        option2_text = menu_font.render("2. Player vs Player", True, WHITE)
        screen.blit(title_text, (250, 150))
        screen.blit(option1_text, (250, 250))
        screen.blit(option2_text, (250, 350))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    game_type = "Player vs AI"
                    paused = False
                    menu_running = False
                elif event.key == pygame.K_2:
                    game_type = "Player vs Player"
                    paused = False
                    menu_running = False

# Reset the ball for a new game
reset_ball()

# Show the initial menu
show_menu()

# Main game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:  # Pause and show menu
                paused = True
                show_menu()

    # Skip game updates if paused
    if paused:
        continue

    # AI control for the left paddle (only if Player vs AI)
    if game_type == "Player vs AI":
        if paddle1.centery < ball.centery and paddle1.bottom < HEIGHT:
            paddle1.y += ai_speed
        if paddle1.centery > ball.centery and paddle1.top > 0:
            paddle1.y -= ai_speed

    # Player controls for the left paddle (only if Player vs Player)
    keys = pygame.key.get_pressed()
    if game_type == "Player vs Player":
        if keys[pygame.K_w] and paddle1.top > 0:
            paddle1.y -= paddle_speed
        if keys[pygame.K_s] and paddle1.bottom < HEIGHT:
            paddle1.y += paddle_speed

    # Player controls for the right paddle
    if keys[pygame.K_UP] and paddle2.top > 0:
        paddle2.y -= paddle_speed
    if keys[pygame.K_DOWN] and paddle2.bottom < HEIGHT:
        paddle2.y += paddle_speed

    # Update ball position
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with top and bottom walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y = -ball_speed_y

    # Ball collision with paddles
    if ball.colliderect(paddle1):
        ball_speed_x = -ball_speed_x * 1.1
        ball.x = paddle1.right

    if ball.colliderect(paddle2):
        ball_speed_x = -ball_speed_x * 1.1
        ball.x = paddle2.left - ball.width

    # Limit the ball speed
    ball_speed_x = max(-max_speed, min(max_speed, ball_speed_x))
    ball_speed_y = max(-max_speed, min(max_speed, ball_speed_y))

    # Ball goes out of bounds (update score and reset ball)
    if ball.left <= 0:
        score2 += 1
        reset_ball()

    if ball.right >= WIDTH:
        score1 += 1
        reset_ball()

    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, paddle1)
    pygame.draw.rect(screen, WHITE, paddle2)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    # Display scores
    score_text1 = font.render(str(score1), True, WHITE)
    score_text2 = font.render(str(score2), True, WHITE)
    screen.blit(score_text1, (320, 10))
    screen.blit(score_text2, (420, 10))

    # Update display
    pygame.display.flip()

    # Frame rate
    pygame.time.Clock().tick(60)

# Quit Pygame
pygame.quit()
