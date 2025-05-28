import chess
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

#--- 1) Helpers ----------------------------------------------------
PIECE_TO_IDX = {
    'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
    'p':6,'n':7,'b':8,'r':9,'q':10,'k':11
}

def fen_to_vector(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    v = np.zeros(12*64, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        idx = PIECE_TO_IDX[piece.symbol()]
        v[idx*64 + sq] = 1
    return v

def parse_first_float(cell) -> float:
    if pd.isna(cell): return 0.0
    s = str(cell).split(',')[0].strip()
    try: return float(s)
    except ValueError: return 0.0

phase_map = {'opening':0, 'middlegame':1, 'endgame':2}

def extract_features(row) -> np.ndarray:
    base = fen_to_vector(row['fen'])
    extras = [
        phase_map.get(row['phase'], 1),
        row.get('material_diff', 0),
        row.get('time_taken', 0),
        parse_first_float(row.get('eval_before_cp', 0)),
        parse_first_float(row.get('eval_after_cp', 0)),
        parse_first_float(row.get('eval_deviations', 0)),
    ]
    return np.hstack([base, extras])


if __name__ == "__main__":
    #--- 2) Load & preprocess --------------------------------------
    df = pd.read_csv("data/positions_labelled.csv")
    df['candidates'] = (
        df['your_moves']
          .str.split(',')
          .apply(lambda lst: [m.strip() for m in lst])
    )
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    #--- 3) Expand training set (3 labels per position) -----------
    X_train, y_train = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train set"):
        feats = extract_features(row)
        for mv in row['candidates']:
            X_train.append(feats)
            y_train.append(mv)
    X_train = np.vstack(X_train)

    # encode string moves → integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    #--- 4) Fit a Random Forest ------------------------------------
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train_enc)

    #--- 5) Prepare test features & “legal‐move” predict/proba -----
    X_test = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Building test set"):
        X_test.append(extract_features(row))
    X_test = np.vstack(X_test)

    # get full probability distribution for each sample
    probs = clf.predict_proba(X_test)
    legal_preds = []

    for i, fen in enumerate(test_df['fen']):
        board = chess.Board(fen)

        # get legal UCI moves, not SAN
        legal_uci = [mv.uci() for mv in board.legal_moves]
        legal_idxs = [np.where(le.classes_ == uci)[0][0]
                    for uci in legal_uci if uci in le.classes_]

        if legal_idxs:
            best_idx = legal_idxs[np.argmax(probs[i, legal_idxs])]
        else:
            best_idx = np.argmax(probs[i])   # rare fallback

        legal_preds.append(le.classes_[best_idx])

    # --- detailed per‐position feedback ---
    for i, (fen, candidates, pred) in enumerate(zip(
        test_df['fen'], test_df['candidates'], legal_preds
    )):
        status = "CORRECT" if pred in candidates else "WRONG"
        print(f"[{status}] idx={i}")
        print(f"  FEN       : {fen}")
        print(f"  Predicted : {pred}")
        print(f"  Candidates: {candidates}\n")

    # overall accuracy
    successes = sum(pred in c for pred, c in zip(legal_preds, test_df['candidates']))
    accuracy = successes / len(test_df)
    print(f"▶ Test accuracy (predicted ∈ candidates): {accuracy*100:.2f}%")

    #--- 6) Save model & encoder ----------------------------------
    joblib.dump(clf, "models/move_predictor_rf.joblib", compress=3)
    joblib.dump(le,  "models/label_encoder.joblib", compress=3)
