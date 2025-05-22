import chess.pgn
import random
import csv
import os

# Config
PGN_PATH = "../data/lichess_db_standard_rated_2013-01.pgn"
OUTPUT_CSV = "../data/positions_extracted.csv"
POSITIONS_TO_EXTRACT = 300

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def classify_phase(board: chess.Board, move_number: int) -> str:
    """
    Classifies position as opening, middlegame, or endgame based on move number and pieces left.

    - Less than 10 moves (both sides) => opening.
    - Both sides combined has 6 or less minor and major pieces => endgame.
    - Otherwise => middlegame.
    """
    if move_number < 10:
        return "opening"

    impactful_pieces = sum(
        1 for piece in board.piece_map().values()
        if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    )

    return "endgame" if impactful_pieces <= 6 else "middlegame"

def get_material_diff(board: chess.Board) -> int:
    """
    Computes material difference between White and Black using standard piece values.

    Positive number means White is ahead, negative implies black is ahead.
    """
    white_material = sum(
        PIECE_VALUES.get(piece.piece_type, 0)
        for piece in board.piece_map().values() if piece.color == chess.WHITE
    )
    black_material = sum(
        PIECE_VALUES.get(piece.piece_type, 0)
        for piece in board.piece_map().values() if piece.color == chess.BLACK
    )
    return white_material - black_material

def extract_positions(pgn_path: str, max_positions: int = 300) -> list[dict]:
    """
    For each game in the PGN file, randomly extract 1 position after move 5.
    
    Every position will have a corresponding FEN, game phase, move number, turn, and material diff.

    Returns:
        A list of dictionaries, each representing a labelled position.
    """
    positions = []
    with open(pgn_path, "r", encoding="utf-8") as pgn:
        game_counter = 0

        while len(positions) < max_positions:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            game_moves = list(game.mainline_moves())
            if len(game_moves) < 10:
                continue

            # Find valid position (not stalemate, checkmate, insufficient material, etc)
            valid_found = False
            for attempt in range(len(game_moves) - 5):
                idx = random.randint(5, len(game_moves) - 1)
                board = game.board()
                for move in game_moves[:idx]:
                    board.push(move)

                if board.is_game_over():
                    continue  # Skip terminal positions

                move_number = idx + 1
                positions.append({
                    "fen": board.fen(),
                    "phase": classify_phase(board, move_number),
                    "move_number": move_number,
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "material_diff": get_material_diff(board)
                })
                valid_found = True
                break  # One valid position per game

            if valid_found and len(positions) >= max_positions:
                break

            game_counter += 1
            if game_counter % 100 == 0:
                print(f"Checked {game_counter} games...")

    return positions

def save_to_csv(positions: list[dict], output_path: str) -> None:
    """
    Saves extracted positions to a CSV file.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "phase", "move_number", "turn", "material_diff"])
        writer.writeheader()
        writer.writerows(positions)

if __name__ == "__main__":
    print("Extracting positions...")
    positions = extract_positions(PGN_PATH, POSITIONS_TO_EXTRACT)
    save_to_csv(positions, OUTPUT_CSV)
    print(f"Saved {len(positions)} positions to {OUTPUT_CSV}")
