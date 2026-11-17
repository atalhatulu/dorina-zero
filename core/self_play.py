"""core/self_play.py — Oyun-Agnostik Self-Play, Simetri Augmentation, Historical Rakip ve Curriculum Motoru."""
import os
import time
import random
import numpy as np
import torch
import multiprocessing as mp

from core.mcts import GenericMCTS, sample_move
from games.base_game import BaseGame
from minimax_ai import get_ai_move
from minimax_fast import get_ai_move_v2
from perfect_solver import get_solver_best_move
from games.othello_minimax import get_othello_ai_move
try:
    from eval_othello_vs_edax import EdaxBot, action_to_gtp, gtp_to_action
except ImportError:
    EdaxBot = None
    action_to_gtp = None
    gtp_to_action = None


def augment_with_horizontal_flip(history, win_player):
    """
    Connect 4 için yatay simetri (horizontal flip).
    Geriye dönük uyumluluk için korunur.
    """
    return augment_data(history, win_player, game_name="connect4")


def augment_data(history, win_player, game_name="connect4"):
    """
    Tüm geçmiş hamleleri simetri dönüşümleriyle çoğaltır:
      - Connect 4: Yatay simetri (horizontal flip) -> 2x veri artırımı
      - Othello: D4 Dihedral simetrisi (4 rotasyon x 2 yansıma) -> 8x veri artırımı (Pass hamlesi 64 korunarak)
    """
    training_data = []
    is_othello = "othello" in game_name.lower()

    for encoded_x, pol, player in history:
        if win_player == 0:
            v = 0.0
        elif player == win_player:
            v = 1.0
        else:
            v = -1.0

        if is_othello:
            # 8x D4 Simetrisi
            # encoded_x: (C, 8, 8), pol: (65,) [64 kare + 1 pass]
            has_pass = (len(pol) == 65)
            pass_val = pol[64] if has_pass else None
            board_pol = pol[:64].reshape(8, 8) if has_pass else pol.reshape(8, 8)

            for k in [0, 1, 2, 3]:
                # 1. Rotasyon (k * 90 derece saat yönünün tersi)
                r_x = np.rot90(encoded_x, k, axes=(1, 2)).copy()
                r_pol = np.rot90(board_pol, k).flatten()
                if has_pass:
                    r_pol = np.concatenate([r_pol, [pass_val]])
                training_data.append((r_x, r_pol.astype(np.float32), v))

                # 2. Yansıma (Horizontal Flip)
                f_x = np.flip(r_x, axis=2).copy()
                f_pol = np.fliplr(r_pol[:64].reshape(8, 8)).flatten()
                if has_pass:
                    f_pol = np.concatenate([f_pol, [pass_val]])
                training_data.append((f_x, f_pol.astype(np.float32), v))
        else:
            # 1. Orijinal örnek
            training_data.append((encoded_x, pol, v))

            # 2. Yatay Simetri (Horizontal Flip)
            flipped_x = encoded_x[:, :, ::-1].copy()
            flipped_pol = pol[::-1].copy()
            training_data.append((flipped_x, flipped_pol, v))

    return training_data


def play_one_game(model, game: BaseGame, opponent_model=None, model_player=1, num_simulations=150,
                  temperature=1.0, temp_threshold=12, dirichlet_epsilon=0.25,
                  dirichlet_alpha=1.0):
    """
    Tek bir self-play veya historical rakibe karşı maç oynar.
    opponent_model verilirse model `model_player` olur, opponent_model karşı oyuncu olur.
    """
    mcts_model = GenericMCTS(model, game=game, num_simulations=num_simulations)
    mcts_opp = GenericMCTS(opponent_model, game=game, num_simulations=num_simulations) if opponent_model else mcts_model

    state = game.get_initial_state()
    current_player = 1
    history = []
    move_count = 0

    while True:
        terminal_val = game.check_terminal_value(state, current_player)
        if terminal_val is not None:
            break

        temp = temperature if move_count < temp_threshold else 0.0
        mcts = mcts_model if current_player == model_player else mcts_opp
        policy = mcts.search(
            state, current_player,
            temperature=temp,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )

        encoded = game.encode_state(state, current_player)
        history.append((encoded, policy.copy(), current_player))

        action = sample_move(policy)
        state = game.get_next_state(state, action, current_player)
        move_count += 1
        current_player = 2 if current_player == 1 else 1

    last_player = 2 if current_player == 1 else 1
    term_from_last = game.check_terminal_value(state, last_player)

    if term_from_last is None or term_from_last == 0.0:
        winner = "draw"
        win_player = 0
    elif term_from_last > 0:
        win_player = last_player
        winner = f"player{win_player}"
    else:
        win_player = 1 if last_player == 2 else 2
        winner = f"player{win_player}"

    training_data = augment_data(history, win_player, game_name=game.name)
    return training_data, winner, move_count


def play_curriculum_game(model, game: BaseGame, model_player=1, minimax_depth=6,
                         num_simulations=150, temperature=1.0, temp_threshold=12,
                         dirichlet_epsilon=0.25, dirichlet_alpha=1.0, opening_moves=None):
    """
    Model vs Minimax curriculum oyunu.
    Tüm win/loss/draw oyunları eşit şekilde ve yatay simetri augmentation ile buffer'a aktarılır.
    """
    mcts = GenericMCTS(model, game=game, num_simulations=num_simulations)
    state = game.get_initial_state()
    current_player = 1
    history = []
    move_count = 0

    if opening_moves:
        for om in opening_moves:
            state = game.get_next_state(state, om, current_player)
            move_count += 1
            current_player = 2 if current_player == 1 else 1

    moves_played = list(opening_moves) if opening_moves else []

    while True:
        terminal_val = game.check_terminal_value(state, current_player)
        if terminal_val is not None:
            break

        encoded = game.encode_state(state, current_player)

        if current_player == model_player:
            temp = temperature if move_count < temp_threshold else 0.0
            policy = mcts.search(
                state, current_player,
                temperature=temp,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon
            )
            action = sample_move(policy)
            history.append((encoded, policy.copy(), current_player))
        else:
            board_copy = game.clone_state(state)
            action = None
            if "othello" in game.name.lower():
                if _worker_args and _worker_args.get("curriculum_mode") == "edax" and EdaxBot is not None:
                    # Edax Bot ile hızlı profesyonel hamle seçimi
                    edax_level = minimax_depth or 1
                    edax = EdaxBot(level=edax_level)
                    try:
                        # O anki renk
                        bot_color = "black" if current_player == 1 else "white"
                        # Daha önce oynanan hamleleri Edax tahtasına replay et
                        replay_state = game.get_initial_state()
                        rp = 1
                        for ma in moves_played:
                            c_str = "black" if rp == 1 else "white"
                            edax.play_move(c_str, action_to_gtp(ma))
                            replay_state = game.get_next_state(replay_state, ma, rp)
                            rp = 2 if rp == 1 else 1
                        gtp_m = edax.genmove(bot_color)
                        action = gtp_to_action(gtp_m)
                    finally:
                        edax.close()
                if action is None:
                    action = get_othello_ai_move(board_copy, depth=minimax_depth, ai_player=current_player)
            else:
                if _worker_args and _worker_args.get("curriculum_mode") == "solver":
                    action, _, _ = get_solver_best_move(moves_played)
                if action is None:
                    if minimax_depth >= 8:
                        action = get_ai_move_v2(board_copy, depth=minimax_depth, ai_player=current_player)
                    else:
                        action = get_ai_move(board_copy, depth=minimax_depth, ai_player=current_player)

            # Bot hamlesi için MCTS süzgeci ve yumuşatılmış politika (Soft Teacher Distribution)
            # %100 hard Dirac etiketi yerine, MCTS dağılımı ile harmanlayarak gradyan patlamasını engeller
            mcts_policy = mcts.search(state, current_player, temperature=1.0, dirichlet_epsilon=0.0)
            policy = 0.3 * mcts_policy
            policy[action] += 0.7
            policy = policy / np.sum(policy)
            history.append((encoded, policy.astype(np.float32), current_player))

        moves_played.append(action)
        state = game.get_next_state(state, action, current_player)
        move_count += 1
        current_player = 2 if current_player == 1 else 1

    last_player = 2 if current_player == 1 else 1
    term_from_last = game.check_terminal_value(state, last_player)

    if term_from_last is None or term_from_last == 0.0:
        win_player = 0
        winner = "draw"
    elif term_from_last > 0:
        win_player = last_player
        winner = "model" if win_player == model_player else "minimax"
    else:
        win_player = 1 if last_player == 2 else 2
        winner = "model" if win_player == model_player else "minimax"

    training_data = augment_data(history, win_player, game_name=game.name)
    return training_data, winner, move_count


# Paralel worker altyapısı
_worker_model = None
_worker_game = None
_worker_hist_models = {}
_worker_args = None


def _init_worker(model_factory, model_state, game_factory, num_simulations,
                 temp_threshold, dirichlet_epsilon, dirichlet_alpha,
                 curriculum_mode, curriculum_depth, historical_states):
    global _worker_model, _worker_game, _worker_hist_models, _worker_args
    torch.set_num_threads(1)
    _worker_model = model_factory()
    _worker_model.load_state_dict(model_state)
    _worker_model.eval()
    _worker_game = game_factory()

    _worker_hist_models = {}
    if historical_states:
        for name, st in historical_states.items():
            hm = model_factory()
            hm.load_state_dict(st)
            hm.eval()
            _worker_hist_models[name] = hm

    _worker_args = {
        "num_simulations": num_simulations,
        "temp_threshold": temp_threshold,
        "dirichlet_epsilon": dirichlet_epsilon,
        "dirichlet_alpha": dirichlet_alpha,
        "curriculum_mode": curriculum_mode,
        "curriculum_depth": curriculum_depth,
    }


def _worker_task(task_spec):
    """task_spec: (game_id, task_type, extra_arg)
    task_type: 'self_play', 'curriculum', 'historical'
    """
    game_id, task_type, extra_arg = task_spec

    if task_type == "curriculum":
        model_player, opening_moves = extra_arg
        data, winner, moves = play_curriculum_game(
            _worker_model,
            _worker_game,
            model_player=model_player,
            minimax_depth=_worker_args["curriculum_depth"],
            num_simulations=_worker_args["num_simulations"],
            temp_threshold=_worker_args["temp_threshold"],
            dirichlet_epsilon=_worker_args["dirichlet_epsilon"],
            dirichlet_alpha=_worker_args["dirichlet_alpha"],
            opening_moves=opening_moves,
        )
        return {"game_id": game_id, "data": data, "winner": winner, "moves": moves, "type": "curriculum"}

    elif task_type == "historical":
        hist_name, model_player = extra_arg
        opp_model = _worker_hist_models.get(hist_name)
        data, winner, moves = play_one_game(
            _worker_model,
            _worker_game,
            opponent_model=opp_model,
            model_player=model_player,
            num_simulations=_worker_args["num_simulations"],
            temp_threshold=_worker_args["temp_threshold"],
            dirichlet_epsilon=_worker_args["dirichlet_epsilon"],
            dirichlet_alpha=_worker_args["dirichlet_alpha"],
        )
        return {"game_id": game_id, "data": data, "winner": winner, "moves": moves, "type": "historical"}

    else:
        data, winner, moves = play_one_game(
            _worker_model,
            _worker_game,
            opponent_model=None,
            num_simulations=_worker_args["num_simulations"],
            temp_threshold=_worker_args["temp_threshold"],
            dirichlet_epsilon=_worker_args["dirichlet_epsilon"],
            dirichlet_alpha=_worker_args["dirichlet_alpha"],
        )
        return {"game_id": game_id, "data": data, "winner": winner, "moves": moves, "type": "self_play"}


def parallel_self_play(model_factory, model, game_factory, num_games=300,
                       num_simulations=150, temp_threshold=12, dirichlet_epsilon=0.25,
                       dirichlet_alpha=1.0, curriculum_mode=None, curriculum_ratio=0.0,
                       curriculum_depth=6, historical_pool=None, historical_ratio=0.0,
                       num_workers=None, verbose=True, iteration=None, status_file=None):
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    # Worker'lar CPU üzerinde çalışır, GPU tensörleri multiprocessing havuzuna aktarılamaz
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Historical modellerin state'lerini topla
    historical_states = {}
    if historical_pool and historical_ratio > 0:
        for p in historical_pool:
            if os.path.exists(p):
                name = os.path.basename(p)
                ckpt = torch.load(p, map_location="cpu")
                historical_states[name] = ckpt.get("model_state", ckpt)

    num_curriculum = int(num_games * curriculum_ratio) if curriculum_mode else 0
    num_historical = int(num_games * historical_ratio) if historical_states else 0
    num_self_play = num_games - num_curriculum - num_historical

    task_specs = []
    # 1. Normal Self-play
    for i in range(num_self_play):
        task_specs.append((i, "self_play", None))

    # 2. Historical Self-play (hem P1 hem P2 olarak donusumlu)
    hist_names = list(historical_states.keys())
    for j in range(num_historical):
        chosen_hist = random.choice(hist_names)
        model_player = 1 if (j % 2 == 0) else 2
        task_specs.append((num_self_play + j, "historical", (chosen_hist, model_player)))

    # 3. Curriculum
    temp_game = game_factory()
    is_othello = "othello" in temp_game.name.lower()
    if is_othello:
        # Othello için rastgele 2-4 ply tarafsız açılış havuzu
        openings_pool = _generate_gating_openings(temp_game, max(1, num_curriculum), min_ply=2, max_ply=4)
        if not openings_pool:
            openings_pool = [None]
    else:
        openings_pool = [
            [1, 1], [4, 2, 6], [0, 4, 1, 5], [5, 4, 3, 1],
            [0, 0], [6, 6], [1, 5], [0, 6], None
        ]

    for k in range(num_curriculum):
        model_player = 1 if (k % 2 == 0) else 2
        chosen_opening = openings_pool[k % len(openings_pool)]
        task_specs.append((num_self_play + num_historical + k, "curriculum", (model_player, chosen_opening)))

    if verbose:
        details = []
        if num_curriculum > 0:
            details.append(f"{num_curriculum} {curriculum_mode} curriculum")
        if num_historical > 0:
            details.append(f"{num_historical} historical vs {hist_names}")
        extra_str = f" (+ {', '.join(details)})" if details else ""
        print(f"  {num_workers} worker ile {num_games} oyun oynanıyor{extra_str}...")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(
            model_factory,
            model_state,
            game_factory,
            num_simulations,
            temp_threshold,
            dirichlet_epsilon,
            dirichlet_alpha,
            curriculum_mode,
            curriculum_depth,
            historical_states,
        ),
    ) as pool:
        results = []
        status_path = status_file or os.path.join("logs_v3", "live_status.json")
        sp_start_time = time.time()
        for idx, r in enumerate(pool.imap_unordered(_worker_task, task_specs, chunksize=2), 1):
            results.append(r)
            if idx % 5 == 0 or idx == len(task_specs):
                try:
                    os.makedirs(os.path.dirname(status_path), exist_ok=True)
                    elapsed = max(0.1, time.time() - sp_start_time)
                    games_per_sec = idx / elapsed
                    remaining = (len(task_specs) - idx) / games_per_sec if games_per_sec > 0 else 0
                    with open(status_path, "w", encoding="utf-8") as sf:
                        import json
                        json.dump({
                            "stage": "self_play",
                            "iteration": iteration,
                            "completed_games": idx,
                            "total_games": len(task_specs),
                            "progress_pct": round(idx / len(task_specs) * 100, 1),
                            "games_per_sec": round(games_per_sec, 2),
                            "elapsed_sec": round(elapsed, 1),
                            "eta_sec": round(remaining, 1),
                            "updated_at": time.time(),
                        }, sf)
                except Exception:
                    pass

    all_data = []
    stats = {
        "player1": 0, "player2": 0, "draw": 0, "total_moves": 0,
        "curriculum_total": num_curriculum,
        "curriculum_model_wins": 0,
        "curriculum_minimax_wins": 0,
        "curriculum_draws": 0,
        "historical_total": num_historical,
        "historical_p1_wins": 0,
        "historical_p2_wins": 0,
        "historical_draws": 0,
    }

    for r in results:
        all_data.extend(r["data"])
        stats["total_moves"] += r["moves"]
        tt = r["type"]
        if tt == "curriculum":
            if r["winner"] == "model":
                stats["curriculum_model_wins"] += 1
            elif r["winner"] == "minimax":
                stats["curriculum_minimax_wins"] += 1
            else:
                stats["curriculum_draws"] += 1
        elif tt == "historical":
            if r["winner"] == "player1":
                stats["historical_p1_wins"] += 1
            elif r["winner"] == "player2":
                stats["historical_p2_wins"] += 1
            else:
                stats["historical_draws"] += 1
        else:
            stats[r["winner"]] += 1

    stats["avg_moves"] = stats["total_moves"] / max(num_games, 1)
    stats["total_samples"] = len(all_data)
    return all_data, stats


_gating_model_a = None
_gating_model_b = None
_gating_game = None
_gating_sims = 150


def _init_gating_worker(model_factory, state_a, state_b, game_factory, num_simulations):
    global _gating_model_a, _gating_model_b, _gating_game, _gating_sims
    torch.set_num_threads(1)
    _gating_model_a = model_factory()
    _gating_model_a.load_state_dict(state_a)
    _gating_model_a.eval()

    _gating_model_b = model_factory()
    _gating_model_b.load_state_dict(state_b)
    _gating_model_b.eval()

    _gating_game = game_factory()
    _gating_sims = num_simulations


def _gating_worker_task(spec):
    opening_moves, a_player = spec
    state = _gating_game.get_initial_state()
    cur_p = 1
    for action in opening_moves:
        state = _gating_game.get_next_state(state, action, cur_p)
        cur_p = 2 if cur_p == 1 else 1

    mcts_a = GenericMCTS(_gating_model_a, game=_gating_game, num_simulations=_gating_sims)
    mcts_b = GenericMCTS(_gating_model_b, game=_gating_game, num_simulations=_gating_sims)

    while True:
        term = _gating_game.check_terminal_value(state, cur_p)
        if term is not None:
            break
        mcts = mcts_a if cur_p == a_player else mcts_b
        policy = mcts.search(state, cur_p, temperature=0.0, dirichlet_epsilon=0.0)
        action = int(np.argmax(policy))
        state = _gating_game.get_next_state(state, action, cur_p)
        cur_p = 2 if cur_p == 1 else 1

    last_p = 2 if cur_p == 1 else 1
    last_term = _gating_game.check_terminal_value(state, last_p)
    if last_term is None or last_term == 0.0:
        return "draw"
    winner_p = last_p if last_term > 0 else (2 if last_p == 1 else 1)
    return "a" if winner_p == a_player else "b"


def _generate_gating_openings(game: BaseGame, num_openings: int, min_ply: int = 2, max_ply: int = 4, seed: int = None):
    rng = random.Random(seed)
    openings = []
    seen = set()
    attempts = 0
    while len(openings) < num_openings and attempts < num_openings * 50:
        attempts += 1
        ply = rng.randint(min_ply, max_ply)
        state = game.get_initial_state()
        cur_p = 1
        moves = []
        terminal = False
        for _ in range(ply):
            valid_mask = game.get_valid_moves(state, cur_p)
            valid_indices = [int(i) for i in np.where(valid_mask)[0]]
            if not valid_indices:
                terminal = True
                break
            action = rng.choice(valid_indices)
            state = game.get_next_state(state, action, cur_p)
            moves.append(action)
            if game.check_terminal_value(state, cur_p) is not None:
                terminal = True
                break
            cur_p = 2 if cur_p == 1 else 1
        t_moves = tuple(moves)
        if not terminal and t_moves not in seen:
            seen.add(t_moves)
            openings.append(moves)
    return openings


def evaluate_gating_parallel(model_factory, model_a, model_b, game_factory,
                             num_games=100, num_simulations=150, num_workers=None, seed=None):
    """
    Rastgele acilislarla (her acilis 2 renk: A=P1 ve A=P2) paralel GenericMCTS gating maci.
    Donus: (a_wins, b_wins, draws)
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    game = game_factory()
    num_openings = max(1, num_games // 2)
    openings = _generate_gating_openings(game, num_openings, min_ply=2, max_ply=4, seed=seed)

    task_specs = []
    for op in openings:
        task_specs.append((op, 1))
        task_specs.append((op, 2))

    state_a = {k: v.cpu() for k, v in model_a.state_dict().items()}
    state_b = {k: v.cpu() for k, v in model_b.state_dict().items()}

    with mp.Pool(
        processes=num_workers,
        initializer=_init_gating_worker,
        initargs=(model_factory, state_a, state_b, game_factory, num_simulations),
    ) as pool:
        results = pool.map(_gating_worker_task, task_specs, chunksize=2)

    a_wins = results.count("a")
    b_wins = results.count("b")
    draws = results.count("draw")
    return a_wins, b_wins, draws
