"""Pygame arayüzü. Canlı değerlendirme çubuğu (evaluation bar) ve analiz göstergeleri ile."""
import sys
import time
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
GRAY = (70, 70, 70)
DARK_GRAY = (40, 40, 40)


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


def draw_header(screen, turn, value=0.0, win_prob=50.0, info_text="", show_eval=True):
    """Üst panel: Sıra göstergesi, Evaluation Bar ve AI Value bilgisi."""
    # Üst alanı temizle
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))

    font_ui = pygame.font.SysFont("monospace", 16, bold=True)
    font_bold = pygame.font.SysFont("monospace", 18, bold=True)

    # 1. Sol Bölüm: Sıra Kimde?
    turn_str = "Sıra: Sen (Kırmızı)" if turn == PLAYER else "Sıra: AI (Sarı)"
    turn_color = RED if turn == PLAYER else YELLOW
    text_turn = font_bold.render(turn_str, True, turn_color)
    screen.blit(text_turn, (15, 20))

    if info_text:
        text_info = font_ui.render(info_text, True, WHITE)
        screen.blit(text_info, (15, 55))

    if show_eval:
        # 2. Orta Bölüm: Evaluation Bar
        bar_w = int(WIDTH * 0.38)
        bar_h = 24
        bar_x = (WIDTH - bar_w) // 2
        bar_y = 20

        # Arka plan çerçevesi
        pygame.draw.rect(screen, DARK_GRAY, (bar_x - 2, bar_y - 2, bar_w + 4, bar_h + 4), border_radius=4)

        # value: -1.0 (İnsan üstün) .. +1.0 (AI üstün)
        # Orta nokta = bar_w // 2
        half_w = bar_w // 2
        center_x = bar_x + half_w

        # Sol taraf kırmızı (İnsan), sağ taraf sarı (AI)
        # Nötr zemin
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_w, bar_h))

        val_clamped = max(-1.0, min(1.0, value))
        if val_clamped < 0:
            # İnsan üstün: ortadan sola kırmızı çubuk
            fill_len = int(abs(val_clamped) * half_w)
            pygame.draw.rect(screen, RED, (center_x - fill_len, bar_y, fill_len, bar_h))
        elif val_clamped > 0:
            # AI üstün: ortadan sağa sarı çubuk
            fill_len = int(val_clamped * half_w)
            pygame.draw.rect(screen, YELLOW, (center_x, bar_y, fill_len, bar_h))

        # Orta çizgi
        pygame.draw.line(screen, WHITE, (center_x, bar_y), (center_x, bar_y + bar_h), 2)

        # 3. Sağ Bölüm: AI Value ve Kazanma İhtimali
        val_sign = "+" if value >= 0 else ""
        txt_val = font_bold.render(f"AI: %{win_prob:.0f} ({val_sign}{value:.2f})", True, YELLOW if value >= 0 else RED)
        screen.blit(txt_val, (WIDTH - txt_val.get_width() - 15, 20))

    pygame.display.update((0, 0, WIDTH, SQUARESIZE))


def run_game(ai_move_fn, ai_name="AI", ai_delay_ms=200, analysis_fn=None, eval_fn=None):
    """
    Oyunu başlatır.
    ai_move_fn(board) -> col: AI hamlesi.
    analysis_fn(board, current_player) -> dict: MCTS analizi.
    eval_fn(board, current_player) -> (value, win_probability): Hızlı NN değerlendirmesi.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"dorina4 - Connect 4 ({ai_name})")
    font = pygame.font.SysFont("monospace", 55, bold=True)
    tip_font = pygame.font.SysFont("monospace", 26)
    clock = pygame.time.Clock()

    board = create_board()
    game_over, turn, message = False, PLAYER, ""

    cur_value = 0.0
    cur_prob = 50.0
    info_msg = ""
    info_expiry = 0.0

    if eval_fn:
        cur_value, cur_prob = eval_fn(board, AI)

    screen.fill(BLACK)
    draw_board(screen, board)
    draw_header(screen, turn, cur_value, cur_prob, info_msg, show_eval=(eval_fn is not None))

    while True:
        player_col = None
        now = time.time()

        if info_msg and now > info_expiry:
            info_msg = ""
            draw_header(screen, turn, cur_value, cur_prob, info_msg, show_eval=(eval_fn is not None))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board = create_board()
                game_over, turn, message = False, PLAYER, ""
                info_msg = ""
                if eval_fn:
                    cur_value, cur_prob = eval_fn(board, AI)
                screen.fill(BLACK)
                draw_board(screen, board)
                draw_header(screen, turn, cur_value, cur_prob, info_msg, show_eval=(eval_fn is not None))
            elif event.type == pygame.MOUSEMOTION and not game_over and turn == PLAYER:
                draw_header(screen, turn, cur_value, cur_prob, info_msg, show_eval=(eval_fn is not None))
                x = event.pos[0]
                col = x // SQUARESIZE
                if 0 <= col < COLS and is_valid_location(board, col):
                    pygame.draw.circle(screen, RED, (x, SQUARESIZE // 2 + 15), RADIUS // 2)
                pygame.display.update((0, 0, WIDTH, SQUARESIZE))
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over and turn == PLAYER:
                col = event.pos[0] // SQUARESIZE
                if 0 <= col < COLS and is_valid_location(board, col):
                    player_col = col

        # İnsan oyuncunun hamlesi
        if not game_over and turn == PLAYER and player_col is not None:
            row = get_next_open_row(board, player_col)
            drop_piece(board, row, player_col, PLAYER)
            draw_board(screen, board)

            if eval_fn:
                cur_value, cur_prob = eval_fn(board, AI)

            if winning_move(board, PLAYER):
                message, game_over = "YOU WIN!", True
            elif not get_valid_locations(board):
                message, game_over = "DRAW!", True
            else:
                turn = AI

            draw_header(screen, turn, cur_value, cur_prob, info_msg, show_eval=(eval_fn is not None))

        # AI Oyuncunun hamlesi
        if not game_over and turn == AI:
            draw_header(screen, turn, cur_value, cur_prob, "AI düşünüyor...", show_eval=(eval_fn is not None))
            pygame.time.wait(ai_delay_ms)

            chosen_col = None
            if analysis_fn:
                analysis = analysis_fn(board, AI)
                chosen_col = analysis.get("best_move")
                cur_value = analysis.get("value", cur_value)
                cur_prob = analysis.get("win_probability", cur_prob)
                pol = analysis.get("policy")
                prob_pct = pol[chosen_col] * 100.0 if pol is not None else cur_prob
                info_msg = f"En iyi: Sütun {chosen_col + 1} (%{prob_pct:.0f} ihtimal)"
                info_expiry = time.time() + 2.0
            else:
                chosen_col = ai_move_fn(board)
                if eval_fn:
                    cur_value, cur_prob = eval_fn(board, AI)

            if chosen_col is not None:
                row = get_next_open_row(board, chosen_col)
                drop_piece(board, row, chosen_col, AI)
                draw_board(screen, board)

                if winning_move(board, AI):
                    message, game_over = "AI WINS!", True
                elif not get_valid_locations(board):
                    message, game_over = "DRAW!", True
                else:
                    turn = PLAYER

            draw_header(screen, turn, cur_value, cur_prob, info_msg, show_eval=(eval_fn is not None))

        # Oyun bitti mesajı
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
