"""Pygame arayüzü. AI fonksiyonu parametre olarak alınır."""
import sys
import pygame

from board import (
    ROWS, COLS, EMPTY, PLAYER, AI,
    create_board, drop_piece, is_valid_location,
    get_next_open_row, winning_move, get_valid_locations,
)

SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE

BLUE = (30, 90, 200)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
YELLOW = (240, 220, 50)
WHITE = (255, 255, 255)


def draw_board(screen, board):
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE,
                (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                (int(c * SQUARESIZE + SQUARESIZE / 2),
                 int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] == PLAYER:
                color = RED
            elif board[r][c] == AI:
                color = YELLOW
            else:
                continue
            pygame.draw.circle(screen, color,
                (int(c * SQUARESIZE + SQUARESIZE / 2),
                 int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


def run_game(ai_move_fn, ai_name="AI", ai_delay_ms=200):
    """
    Oyunu başlatır.
    ai_move_fn(board) -> col: AI'ın hamlesini döndüren fonksiyon.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"dorina4 - Connect 4 ({ai_name})")
    font = pygame.font.SysFont("monospace", 55, bold=True)
    tip_font = pygame.font.SysFont("monospace", 26)
    clock = pygame.time.Clock()

    board = create_board()
    game_over, turn, message = False, PLAYER, ""

    screen.fill(BLACK)
    draw_board(screen, board)

    while True:
        player_col = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board = create_board()
                game_over, turn, message = False, PLAYER, ""
                screen.fill(BLACK)
                draw_board(screen, board)
            elif event.type == pygame.MOUSEMOTION and not game_over and turn == PLAYER:
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                x = event.pos[0]
                col = x // SQUARESIZE
                if 0 <= col < COLS and is_valid_location(board, col):
                    pygame.draw.circle(screen, RED, (x, SQUARESIZE // 2), RADIUS)
                pygame.display.update()
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over and turn == PLAYER:
                col = event.pos[0] // SQUARESIZE
                if 0 <= col < COLS and is_valid_location(board, col):
                    player_col = col

        if not game_over and turn == PLAYER and player_col is not None:
            row = get_next_open_row(board, player_col)
            drop_piece(board, row, player_col, PLAYER)
            pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
            draw_board(screen, board)
            if winning_move(board, PLAYER):
                message, game_over = "YOU WIN!", True
            elif not get_valid_locations(board):
                message, game_over = "DRAW!", True
            else:
                turn = AI

        if not game_over and turn == AI:
            pygame.time.wait(ai_delay_ms)
            col = ai_move_fn(board)
            if col is not None:
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI)
                draw_board(screen, board)
                if winning_move(board, AI):
                    message, game_over = "AI WINS!", True
                elif not get_valid_locations(board):
                    message, game_over = "DRAW!", True
                else:
                    turn = PLAYER

        if game_over and message:
            overlay = pygame.Surface((WIDTH, SQUARESIZE * 2), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            screen.blit(overlay, (0, SQUARESIZE * 3))
            text = font.render(message, True, WHITE)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, SQUARESIZE * 3 + 20))
            tip = tip_font.render("Press R to restart", True, WHITE)
            screen.blit(tip, (WIDTH // 2 - tip.get_width() // 2, SQUARESIZE * 3 + 95))
            pygame.display.update()

        clock.tick(30)