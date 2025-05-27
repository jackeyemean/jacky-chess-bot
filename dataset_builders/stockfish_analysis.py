import chess
import chess.engine
import pandas as pd
from tqdm import tqdm  # progress bar

LABELLED_POSITIONS_PATH = "data/positions_labelled.csv"
STOCKFISH_PATH         = "C:\\Users\\jacky\\repos\\stockfish\\stockfish-windows-x86-64-avx2.exe"

df = pd.read_csv(LABELLED_POSITIONS_PATH)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# accumulators for every position
centipawns_before = []
centipawns_after = []
mate_distance_before = []
mate_distance_after = []
predicted_best_moves = []
evaluation_deviations = []

# process each row (aka position)
for _, row in tqdm(df.iterrows(), total=len(df), desc="Stockfish eval"):
    board = chess.Board(row["fen"])
    # now contains three comma-delimited moves
    candidate_moves = row["your_moves"].split(",")

    # analysis before move (same for all three)
    try:
        analysis_before = engine.analyse(board, chess.engine.Limit(depth=20))
        score_before = analysis_before["score"].white()
        cp_before = score_before.score(mate_score=10_000)
        mate_before = score_before.mate()

        pv = analysis_before.get("pv", [])  # pv stands for "principal variation"
        best_move_pred = pv[0].uci() if pv else None
    except:
        centipawns_before.append(None)
        centipawns_after.append(",".join([ "" for _ in candidate_moves ]))
        mate_distance_before.append(None)
        mate_distance_after.append(",".join([ "" for _ in candidate_moves ]))
        predicted_best_moves.append(best_move_pred if 'best_move_pred' in locals() else None)
        evaluation_deviations.append(",".join([ "" for _ in candidate_moves ]))
        continue

    # save single-before values
    centipawns_before.append(cp_before)
    mate_distance_before.append(mate_before)
    predicted_best_moves.append(best_move_pred)

    after_cps = []
    after_mates = []
    after_devs = []

    for mv in candidate_moves:
        b2 = board.copy()
        b2.push(chess.Move.from_uci(mv))

        try:
            analysis_after = engine.analyse(b2, chess.engine.Limit(depth=20))
            score_after = analysis_after["score"].white()
            cp_after = score_after.score(mate_score=10_000)
            mate_after = score_after.mate()
        except:
            after_cps.append(None)
            after_mates.append(None)
            after_devs.append(None)
            continue

        after_cps.append(cp_after)
        after_mates.append(mate_after)

        # edge case handling
        if b2.is_game_over():
            result = b2.result()
            if result == "1/2-1/2":
                deviation = abs(cp_before)
            else:
                deviation = 0
        elif mate_before is not None and mate_after is not None:
            deviation = abs(mate_before - mate_after)
        elif mate_before is not None and mate_after is None:
            deviation = 10_000 + abs(cp_after)
        elif mate_before is None and mate_after is not None:
            deviation = 10_000 + abs(cp_before)
        else:
            deviation = abs(cp_after - cp_before)

        after_devs.append(deviation)

    # comma-delimit everything
    centipawns_after.append(",".join(str(x) for x in after_cps))
    mate_distance_after.append(",".join(str(x) for x in after_mates))
    evaluation_deviations.append(",".join(str(x) for x in after_devs))

engine.quit()

df["eval_before_cp"] = centipawns_before
df["eval_after_cp"] = centipawns_after
df["mate_before_dist"] = mate_distance_before
df["mate_after_dist"] = mate_distance_after
df["best_stockfish"] = predicted_best_moves
df["eval_deviations"] = evaluation_deviations

df.to_csv(LABELLED_POSITIONS_PATH, index=False)
print("Updated positions_labelled.csv with Stockfish evaluations and best moves.")
