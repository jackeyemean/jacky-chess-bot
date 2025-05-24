import chess
import chess.engine
import pandas as pd

LABELLED_POSITIONS_PATH = "data/positions_labelled.csv"
STOCKFISH_PATH = "C:\\Users\\jacky\\repos\\stockfish\\stockfish-windows-x86-64-avx2.exe"

df = pd.read_csv(LABELLED_POSITIONS_PATH)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# accumulators for every position
centipawns_before = []
centipawns_after = []
mate_distance_before = []
mate_distance_after = []
predicted_best_moves = []
evaluation_deviation = []

# process each row (aka position)
for _, row in df.iterrows():
    board = chess.Board(row["fen"])
    move = chess.Move.from_uci(row["your_move"])

    # analysis before move
    try:
        analysis_before = engine.analyse(board, chess.engine.Limit(time=0.1))
        score_before = analysis_before["score"].white()
        cp_before = score_before.score(mate_score=10_000)
        mate_before = score_before.mate()

        pv = analysis_before.get("pv", [])  # pv stands for "principal variation" (aka engine's current best line of play)
        best_move_pred = pv[0].uci() if pv else None
    except:
        centipawns_before.append(None)
        centipawns_after.append(None)
        mate_distance_before.append(None)
        mate_distance_after.append(None)
        predicted_best_moves.append(None)
        evaluation_deviation.append(None)
        continue

    # save to accumulators
    centipawns_before.append(cp_before)
    mate_distance_before.append(mate_before)
    predicted_best_moves.append(best_move_pred)

    # apply move
    board.push(move)

    # analysis after move
    try:
        analysis_after = engine.analyse(board, chess.engine.Limit(time=0.1))
        score_after = analysis_after["score"].white()
        cp_after = score_after.score(mate_score=10_000)
        mate_after = score_after.mate()
    except:
        centipawns_after.append(None)
        mate_distance_after.append(None)
        evaluation_deviation.append(None)
        continue

    # save to accumulators
    centipawns_after.append(cp_after)
    mate_distance_after.append(mate_after)
    
    # edge case handling
    if board.is_game_over():
        # game ended after this move: draw or delivered mate
        result = board.result()  # "1-0", "0-1", or "1/2-1/2"
        if result == "1/2-1/2":
            # draw — you missed whatever edge you had
            deviation = abs(cp_before)
        else:
            # you delivered the win/mate — no penalty
            deviation = 0

    elif mate_before is not None and mate_after is not None:
        # checkmate still possible in both positions, measure change in mate distance
        deviation = abs(mate_before - mate_after)

    elif mate_before is not None and mate_after is None:
        # lost a forced mate, heavy penalty
        deviation = 10_000 + abs(cp_after)

    elif mate_before is None and mate_after is not None:
        # walked into a forced mate, heavy penalty
        deviation = 10_000 + abs(cp_before)

    else:
        # no forced mate in either: simple centipawn swing
        deviation = abs(cp_after - cp_before)
    
    evaluation_deviation.append(deviation)

engine.quit()

df["eval_before_cp"] = centipawns_before
df["eval_after_cp"] = centipawns_after
df["mate_before_dist"] = mate_distance_before
df["mate_after_dist"] = mate_distance_after
df["best_stockfish"] = predicted_best_moves
df["eval_deviation"] = evaluation_deviation

df.to_csv(LABELLED_POSITIONS_PATH, index=False)
print("Updated positions_labelled.csv with Stockfish evaluations and best move.")
