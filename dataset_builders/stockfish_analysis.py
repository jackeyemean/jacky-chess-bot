import chess
import chess.engine
import pandas as pd

# Config
LABELLED_PATH = "../data/positions_labelled.csv"
STOCKFISH_PATH = "../stockfish/stockfish-windows-x86-64-avx2.exe"

# Load previously labelled positions
df = pd.read_csv(LABELLED_PATH)

# Start stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Accumulators
eval_before = []
eval_after = []
deviations = []
best_moves = []

# Process each row (aka position)
for idx, row in df.iterrows():
    board = chess.Board(row["fen"])
    move = chess.Move.from_uci(row["your_move"])

    try:
        # stockfish evaluation before move
        info_before = engine.analyse(board, chess.engine.Limit(time=0.1))
        score_before = info_before["score"].white()
        eval_before_cp = score_before.score(mate_score=1000)

        # stockfish best move
        best_move = info_before["pv"][0].uci()
        best_moves.append(best_move)

    except:
        eval_before.append(None)
        eval_after.append(None)
        deviations.append(None)
        best_moves.append(None)
        continue

    # Apply player's move
    board.push(move)

    try:
        # stockfish eval after move
        info_after = engine.analyse(board, chess.engine.Limit(time=0.1))
        score_after = info_after["score"].white()
        eval_after_cp = score_after.score(mate_score=1000)
    except:
        eval_before.append(eval_before_cp)
        eval_after.append(None)
        deviations.append(None)
        continue
    
    # Save evals to accumulators
    eval_before.append(eval_before_cp)
    eval_after.append(eval_after_cp)

    # Calculate deviation in evaluation
    # If there was a checkmate possible
    if score_before.is_mate():
        # If there is still a checkmate possible / game ended
        if score_after.is_mate():
            deviations.append(0)
        # If there is no more checkmate possible
        else:
            deviations.append(1000 - eval_after_cp)
    else:
        deviations.append(abs(eval_before_cp - eval_after_cp))

engine.quit()

df["eval_before"] = eval_before
df["eval_after"] = eval_after
df["best_move_deviation"] = deviations
df["best_move"] = best_moves

df.to_csv(LABELLED_PATH, index=False)
print("Updated positions_labelled.csv with Stockfish evaluations and best move.")
