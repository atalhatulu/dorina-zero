import pygame
import sys
import math
import random

# --- Sabitler ---
ROWS, COLS = 6, 7
SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE

BLUE    = (30, 90, 200)
BLACK   = (0, 0, 0)
RED     = (220, 50, 50)
YELLOW  = (240, 220, 50)
WHITE   = (255, 255, 255)

PLAYER, AI, EMPTY = 1, 2, 0

# --- Board ---
def create_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == EMPTY

def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r

def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]

def winning_move(board, piece):
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c + i] == piece for i in range(4)): return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)): return True
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)): return True
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i][c + i] == piece for i in range(4)): return True
    return False

# --- AI ---
def evaluate_window(window, piece):
    score = 0
    opp = AI if piece == PLAYER else PLAYER
    if window.count(piece) == 4: score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1: score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2: score += 2
    if window.count(opp) == 3 and window.count(EMPTY) == 1: score -= 4
    return score

def score_position(board, piece):
    score = 0
    center = [board[r][COLS // 2] for r in range(ROWS)]
    score += center.count(piece) * 3
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += evaluate_window(board[r][c:c + 4], piece)
    for c in range(COLS):
        col_arr = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            score += evaluate_window(col_arr[r:r + 4], piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += evaluate_window([board[r + i][c + i] for i in range(4)], piece)
            score += evaluate_window([board[r + 3 - i][c + i] for i in range(4)], piece)
    return score

def minimax(board, depth, alpha, beta, maximizing):
    valid = get_valid_locations(board)
    terminal = winning_move(board, AI) or winning_move(board, PLAYER) or not valid
    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, AI): return None, 1_000_000
            if winning_move(board, PLAYER): return None, -1_000_000
            return None, 0
        return None, score_position(board, AI)

    if maximizing:
        value, column = -math.inf, random.choice(valid)
        for col in valid:
            row = get_next_open_row(board, col)
            b = [r[:] for r in board]
            drop_piece(b, row, col, AI)
            s = minimax(b, depth - 1, alpha, beta, False)[1]
            if s > value: value, column = s, col
            alpha = max(alpha, value)
            if alpha >= beta: break
        return column, value
    else:
        value, column = math.inf, random.choice(valid)
        for col in valid:
            row = get_next_open_row(board, col)
            b = [r[:] for r in board]
            drop_piece(b, row, col, PLAYER)
            s = minimax(b, depth - 1, alpha, beta, True)[1]
            if s < value: value, column = s, col
            beta = min(beta, value)
            if alpha >= beta: break
        return column, value

# --- Çizim ---
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
            if board[r][c] == PLAYER: color = RED
            elif board[r][c] == AI: color = YELLOW
            else: continue
            pygame.draw.circle(screen, color,
                (int(c * SQUARESIZE + SQUARESIZE / 2),
                 int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

# --- Ana Döngü ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("dorina4 - Connect 4")
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
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board = create_board()
                game_over, turn, message = False, PLAYER, ""
                screen.fill(BLACK); draw_board(screen, board)
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
            pygame.time.wait(200)
            col, _ = minimax(board, 4, -math.inf, math.inf, True)
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

if __name__ == "__main__":
    main()
