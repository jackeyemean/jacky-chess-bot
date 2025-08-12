import pygame
import pandas as pd
import os
import chess
import time

INPUT_CSV = "data/positions_extracted.csv"
OUTPUT_CSV = "data/positions_labelled.csv"
ASSETS_DIR = "assets"
SQUARE_SIZE = 80
WINDOW_SIZE = SQUARE_SIZE * 8
INFO_HEIGHT = 40
TIMER_HEIGHT = 30

WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 43)
TEXT_COLOR = (20, 20, 20)
LEGAL_MOVE_DOT = (0, 255, 0)
BUTTON_COLOR = (200, 0, 0)
BUTTON_TEXT_COLOR = (255, 255, 255)
PROMOTION_COLOR = (100, 100, 200)

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + INFO_HEIGHT + TIMER_HEIGHT))
pygame.display.set_caption("Chess Labeler")
font = pygame.font.SysFont(None, 32)
small_font = pygame.font.SysFont(None, 24)

df = pd.read_csv(INPUT_CSV)
labeled = []
index = 0
if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
    labeled_df = pd.read_csv(OUTPUT_CSV)
    labeled = labeled_df.to_dict("records")
    index = len(labeled)

def load_piece_images() -> dict:
    '''
    - loads and scales chess piece images
    - returns a dict mapping piece codes (e.g., 'wP', 'bN') to scaled pygame.Surface objects.
    '''
    images = {}
    for piece in "PNBRQK":
        for color in "wb":
            filename = f"{color}{piece}.png"
            path = os.path.join(ASSETS_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing image: {path}")
            img = pygame.image.load(path).convert_alpha()
            images[f"{color}{piece}"] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
    return images

piece_images = load_piece_images()

def draw_board(board: chess.Board, from_sq: int = None, legal_moves: list = [], message: str = "", time_elapsed: float = None, show_promotion: bool = False) -> None:
    """
    renders chessboard and UI elements
    """
    flip = not board.turn
    screen.fill((255, 255, 255))

    if time_elapsed is not None:
        timer_text = f"{time_elapsed:.1f}s"
        screen.blit(font.render(timer_text, True, TEXT_COLOR), (10, 5))

    for rank in range(8):
        for file in range(8):
            display_rank = 7 - rank if not flip else rank
            display_file = file if not flip else 7 - file
            square = chess.square(display_file, display_rank)

            color = WHITE if (rank + file) % 2 == 0 else BLACK
            if square == from_sq:
                color = HIGHLIGHT
            rect = pygame.Rect(file*SQUARE_SIZE, TIMER_HEIGHT + rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

            piece = board.piece_at(square)
            if piece:
                color_code = 'w' if piece.color == chess.WHITE else 'b'
                img = piece_images[f"{color_code}{piece.symbol().upper()}"]
                screen.blit(img, (file*SQUARE_SIZE, TIMER_HEIGHT + rank*SQUARE_SIZE))

            if square in legal_moves:
                center = (file * SQUARE_SIZE + SQUARE_SIZE // 2, TIMER_HEIGHT + rank * SQUARE_SIZE + SQUARE_SIZE // 2)
                pygame.draw.circle(screen, LEGAL_MOVE_DOT, center, 10)

    pygame.draw.rect(screen, (220, 220, 220), (0, WINDOW_SIZE + TIMER_HEIGHT, WINDOW_SIZE, INFO_HEIGHT))
    turn_text = "White to move" if board.turn else "Black to move"
    screen.blit(font.render(f"{turn_text}  {message}", True, TEXT_COLOR), (10, WINDOW_SIZE + TIMER_HEIGHT + 5))

    pygame.draw.rect(screen, BUTTON_COLOR, (WINDOW_SIZE - 130, WINDOW_SIZE + TIMER_HEIGHT + 5, 120, 30))
    screen.blit(font.render("← Go Back", True, BUTTON_TEXT_COLOR), (WINDOW_SIZE - 120, WINDOW_SIZE + TIMER_HEIGHT + 10))

    # Draw promotion selection if needed
    if show_promotion:
        promotion_y = WINDOW_SIZE // 2 - 100
        pygame.draw.rect(screen, PROMOTION_COLOR, (WINDOW_SIZE // 2 - 100, promotion_y, 200, 80))
        pygame.draw.rect(screen, (255, 255, 255), (WINDOW_SIZE // 2 - 90, promotion_y + 10, 180, 60))
        
        # Draw promotion pieces
        pieces = ['Q', 'R', 'B', 'N']
        for i, piece in enumerate(pieces):
            x = WINDOW_SIZE // 2 - 80 + i * 40
            y = promotion_y + 20
            pygame.draw.rect(screen, (200, 200, 200), (x, y, 30, 30))
            screen.blit(small_font.render(piece, True, TEXT_COLOR), (x + 10, y + 5))

    pygame.display.flip()

def get_square_from_mouse(pos: tuple, flip: bool = False) -> int:
    '''
    - converts mouse (x, y) tuple position to chess square index
    - also has to check if board is flipped (ie playing as black)
    '''
    file = pos[0] // SQUARE_SIZE
    rank = 7 - ((pos[1] - TIMER_HEIGHT) // SQUARE_SIZE)
    if flip:
        file = 7 - file
        rank = 7 - rank
    return chess.square(file, rank)

def clicked_go_back(pos: tuple) -> bool:
    '''
    checks if back button was selected
    '''
    x, y = pos
    return WINDOW_SIZE - 130 <= x <= WINDOW_SIZE - 10 and WINDOW_SIZE + TIMER_HEIGHT + 5 <= y <= WINDOW_SIZE + TIMER_HEIGHT + 35

def get_promotion_piece_from_mouse(pos: tuple) -> int:
    '''
    gets promotion piece from mouse click on promotion selection
    '''
    x, y = pos
    promotion_y = WINDOW_SIZE // 2 - 100
    
    if promotion_y + 10 <= y <= promotion_y + 70:
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for i, piece in enumerate(pieces):
            piece_x = WINDOW_SIZE // 2 - 80 + i * 40
            if piece_x <= x <= piece_x + 30:
                return piece
    return None

def is_promotion_move(board: chess.Board, from_square: int, to_square: int) -> bool:
    '''
    checks if a move is a pawn promotion
    '''
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_square)
        return rank in [0, 7]  # Pawn reaching 8th or 1st rank
    return False

def is_castling_move(board: chess.Board, from_square: int, to_square: int) -> bool:
    '''
    checks if a move is castling
    '''
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.KING:
        # King moving 2 squares horizontally
        from_file = chess.square_file(from_square)
        to_file = chess.square_file(to_square)
        return abs(from_file - to_file) == 2
    return False

'''
MAIN LOOP
- playing through positions
- collecting user move labels
'''
running = True
while running and index < len(df):
    row = df.iloc[index]
    board = chess.Board(row["fen"])
    move_made = False
    message = ""
    from_square = None
    legal_squares = []
    # collect up to 3 moves
    candidate_moves = []
    start_time = time.time()
    waiting_for_promotion = False
    promotion_from = None
    promotion_to = None

    while not move_made:
        elapsed = round(time.time() - start_time, 1)
        # show how many moves so far
        msg = f"Moves inputted: {len(candidate_moves)}/3"
        draw_board(board, from_sq=from_square, legal_moves=legal_squares, message=msg, time_elapsed=elapsed, show_promotion=waiting_for_promotion)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                move_made = True

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if clicked_go_back(event.pos):
                    if index > 0 and len(labeled) > 0:
                        index -= 1
                        labeled.pop()
                        pd.DataFrame(labeled).to_csv(OUTPUT_CSV, index=False)
                        row = df.iloc[index]
                        board = chess.Board(row["fen"])
                        message = "Already at start of data"
                        # reset for this position
                        candidate_moves = []
                        from_square = None
                        legal_squares = []
                        start_time = time.time()
                        waiting_for_promotion = False
                    else:
                        message = "Already at start of data"
                    continue

                # Handle promotion piece selection
                if waiting_for_promotion:
                    promotion_piece = get_promotion_piece_from_mouse(event.pos)
                    if promotion_piece is not None:
                        # Create the promotion move
                        move = chess.Move(promotion_from, promotion_to, promotion=promotion_piece)
                        candidate_moves.append(move.uci())
                        
                        # Reset for next move
                        from_square = None
                        legal_squares = []
                        waiting_for_promotion = False
                        promotion_from = None
                        promotion_to = None
                        message = ""
                        
                        # Check if we have 3 moves
                        if len(candidate_moves) == 3:
                            total_time = round(time.time() - start_time, 2)
                            labeled.append({
                                "fen": row["fen"],
                                "phase": row["phase"],
                                "move_number": row["move_number"],
                                "turn": row["turn"],
                                "material_diff": row["material_diff"],
                                "your_moves": ",".join(candidate_moves),
                                "time_taken": total_time
                            })
                            pd.DataFrame(labeled).to_csv(OUTPUT_CSV, index=False)
                            index += 1
                            move_made = True
                    continue

                flip = not board.turn
                clicked_square = get_square_from_mouse(event.pos, flip=flip)

                if from_square is None:
                    if board.piece_at(clicked_square) and board.color_at(clicked_square) == board.turn:
                        from_square = clicked_square
                        legal_squares = [m.to_square for m in board.legal_moves if m.from_square == from_square]
                        message = ""
                    else:
                        message = "Invalid piece"
                else:
                    to_square = clicked_square
                    if to_square == from_square:
                        message = "Selection cancelled"
                        from_square = None
                        legal_squares = []
                        continue

                    # Check if this is a promotion move
                    if is_promotion_move(board, from_square, to_square):
                        waiting_for_promotion = True
                        promotion_from = from_square
                        promotion_to = to_square
                        message = "Select promotion piece (Q/R/B/N)"
                        continue

                    # Create the move
                    move = chess.Move(from_square, to_square)
                    if move in board.legal_moves:
                        candidate_moves.append(move.uci())
                        # reset selection for next candidate
                        from_square = None
                        legal_squares = []
                        message = ""
                        # once we have 3, finalize this position
                        if len(candidate_moves) == 3:
                            total_time = round(time.time() - start_time, 2)
                            labeled.append({
                                "fen": row["fen"],
                                "phase": row["phase"],
                                "move_number": row["move_number"],
                                "turn": row["turn"],
                                "material_diff": row["material_diff"],
                                "your_moves": ",".join(candidate_moves),
                                "time_taken": total_time
                            })
                            pd.DataFrame(labeled).to_csv(OUTPUT_CSV, index=False)
                            index += 1
                            move_made = True
                    else:
                        message = "Illegal move"
                        from_square = None
                        legal_squares = []

pygame.quit()
