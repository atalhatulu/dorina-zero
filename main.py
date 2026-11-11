"""dorina4 - Connect 4. Giriş noktası."""
from functools import partial
from minimax_ai import get_ai_move
from gui import run_game


if __name__ == "__main__":
    # Şimdilik baseline minimax AI ile oyna
    ai_fn = partial(get_ai_move, depth=4)
    run_game(ai_fn, ai_name="Minimax (depth 4)")
