import chess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Config
LABELLED_CSV = "data/positions_labelled.csv"
MODEL_PATH = "models/move_predictor_rf.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

# 1) load labelled data
df = pd.read_csv(LABELLED_CSV)

# 2) helper: map piece to plane index
PIECE_TO_IDX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_vector(fen: str) -> np.ndarray:
    """
    Turn a FEN string into a 768-dimensional binary vector:
      12 piece-types × 64 squares.
    """
    board = chess.Board(fen)
    vec = np.zeros(12 * 64, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        code = piece.symbol()
        idx = PIECE_TO_IDX[code]
        vec[idx * 64 + sq] = 1
    return vec

# 3) build feature matrix
#    - board vectors
#    - game phase (opening=0, middlegame=1, endgame=2)
#    - material difference
feature_list = []
for _, row in df.iterrows():
    base = fen_to_vector(row["fen"])
    extras = [
        {"opening": 0, "middlegame": 1, "endgame": 2}[row["phase"]],
        row["material_diff"]
    ]
    feature_list.append(np.hstack([base, extras]))
X = np.vstack(feature_list)

# 4) encode target moves
#    We take the first of the comma-delimited your_moves list as the primary label.
moves = df["your_moves"].str.split(",", expand=True)[0]
le = LabelEncoder()
y = le.fit_transform(moves)

# 5) train/test split (by position)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) fit a Random Forest
clf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# 7) evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"▶ Test accuracy: {acc * 100:.2f}%")

# 8) save model + encoder
joblib.dump(clf, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Label encoder saved to {ENCODER_PATH}")
