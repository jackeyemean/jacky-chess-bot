import chess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1) load labelled data
df = pd.read_csv("data/positions_labelled.csv")

# 2) helper: map piece to plane index
PIECE_TO_IDX = {
    'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
    'p':6,'n':7,'b':8,'r':9,'q':10,'k':11
}

def fen_to_vector(fen: str) -> np.ndarray:
    """
    Turn a FEN into a 768-dim binary vector:
      12 piece-types × 64 squares
    """
    board = chess.Board(fen)
    vec = np.zeros(12*64, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        code = piece.symbol()
        idx = PIECE_TO_IDX[code]
        vec[idx*64 + sq] = 1
    return vec

# 3) build feature matrix
feature_list = []
for _, row in df.iterrows():
    base = fen_to_vector(row["fen"])
    extras = [
        {"opening":0,"middlegame":1,"endgame":2}[row["phase"]],
        row["material_diff"],
        row.get("time_taken", 0),
        row.get("eval_before_cp", 0),
        row.get("eval_after_cp", 0),
        row.get("best_move_deviation", 0),
    ]
    feature_list.append(np.hstack([base, extras]))
X = np.vstack(feature_list)

# 4) encode target moves (we train on the first of your three)
le = LabelEncoder()
first_moves = df["your_moves"].str.split(",").str[0]
y = le.fit_transform(first_moves)

# 5) split into train/test
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=42
)

# 6) fit a Random Forest
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# 7) evaluate top-3 success rate
y_pred = clf.predict(X_test)
y_pred_moves = le.inverse_transform(y_pred)

# pull out the three candidate moves from the hold-out set
candidate_lists = df_test["your_moves"].str.split(",")

# count how many times our prediction is in the 3 candidates
hits = sum(
    pred in candidates
    for pred, candidates in zip(y_pred_moves, candidate_lists)
)
top3_success = hits / len(y_pred_moves)
print(f"▶ Top-3 success rate: {top3_success*100:.2f}%")

# 8) save model + encoder
joblib.dump(clf,    "models/move_predictor_rf.joblib")
joblib.dump(le,     "models/label_encoder.joblib")
