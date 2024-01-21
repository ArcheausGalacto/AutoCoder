# filename: calculator_app.py
import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((300, 400))
pygame.display.set_caption("Calculator")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)

# Define fonts
font = pygame.font.Font(None, 36)

# Define button properties
button_width = 60
button_height = 40
button_margin = 10
buttons = [
    {'label': '7', 'pos': (10, 100)},
    {'label': '8', 'pos': (80, 100)},
    {'label': '9', 'pos': (150, 100)},
    {'label': '/', 'pos': (220, 100)},
    {'label': '4', 'pos': (10, 150)},
    {'label': '5', 'pos': (80, 150)},
    {'label': '6', 'pos': (150, 150)},
    {'label': '*', 'pos': (220, 150)},
    {'label': '1', 'pos': (10, 200)},
    {'label': '2', 'pos': (80, 200)},
    {'label': '3', 'pos': (150, 200)},
    {'label': '-', 'pos': (220, 200)},
    {'label': '0', 'pos': (10, 250)},
    {'label': '.', 'pos': (80, 250)},
    {'label': '=', 'pos': (150, 250)},
    {'label': '+', 'pos': (220, 250)},
]

# Calculator state
current_input = ""
operator = ""
operand1 = ""
operand2 = ""

# Main loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in buttons:
                x, y = button['pos']
                if x <= mouse_pos[0] <= x + button_width and y <= mouse_pos[1] <= y + button_height:
                    label = button['label']
                    if label in '0123456789':
                        current_input += label
                    elif label in '+-*/':
                        if current_input:
                            operand1 = current_input
                            current_input = ""
                        operator = label
                    elif label == '=' and operand1 and current_input:
                        operand2 = current_input
                        try:
                            result = str(eval(operand1 + operator + operand2))
                        except ZeroDivisionError:
                            result = "Error"
                        operand1 = result
                        current_input = ""
                        operator = ""
                    elif label == '.' and '.' not in current_input:
                        current_input += label
                    break

    # Drawing
    screen.fill(WHITE)
    for button in buttons:
        x, y = button['pos']
        pygame.draw.rect(screen, GRAY, (x, y, button_width, button_height))
        text_surface = font.render(button['label'], True, BLACK)
        screen.blit(text_surface, (x + (button_width - text_surface.get_width()) // 2, y + (button_height - text_surface.get_height()) // 2))

    # Display current input
    if current_input:
        display_text = current_input
    elif operand1:
        display_text = operand1
    else:
        display_text = "0"
    text_surface = font.render(display_text, True, BLACK)
    screen.blit(text_surface, (10, 50))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
