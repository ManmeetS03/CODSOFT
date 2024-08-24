import math
board = ['_'] * 9

def task2_tictactoe(board):
    for i in range(0, 9, 3):
        print('----[' + board[i] + '|' + board[i+1] + '|' + board[i+2] + ']----')
    print()

def check_winner(board, player):
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for combo in winning_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

def is_board_full(board):
    return all(cell != '_' for cell in board)

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, AI, YOU):
    if check_winner(board, AI):
        return 1
    elif check_winner(board, YOU):
        return -1
    elif is_board_full(board):
        return 0
    
    if maximizing_player:
        max_eval = -math.inf
        for i in range(9):
            if board[i] == '_':
                board[i] = AI
                eval = minimax_alpha_beta(board, depth + 1, alpha, beta, False, AI, YOU)
                board[i] = '_'
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(9):
            if board[i] == '_':
                board[i] = YOU
                eval = minimax_alpha_beta(board, depth + 1, alpha, beta, True, AI, YOU)
                board[i] = '_'
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

def find_best_move(board, AI, YOU):
    best_move = -1
    best_eval = -math.inf
    for i in range(9):
        if board[i] == '_':
            board[i] = AI
            eval = minimax_alpha_beta(board, 0, -math.inf, math.inf, False, AI, YOU)
            board[i] = '_'
            if eval > best_eval:
                best_eval = eval
                best_move = i
    return best_move

def main():
    YOU = input("Choose your marker (X/O): ").upper()
    while YOU not in ['X', 'O']:
        YOU = input("Please choose either X or O: ").upper()
    
    AI = 'O' if YOU == 'X' else 'X'
    
    while True:
        task2_tictactoe(board)
        
        move = int(input("Select your choice (1-9): ")) - 1
        if move < 0 or move > 9:
            print("Invalid Move. Please Choose a number between 1 and 9.")
            continue
        
        if board[move] == '_':
            board[move] = YOU
            if check_winner(board, YOU):
                task2_tictactoe(board)
                print("You win :D")
                break
            elif is_board_full(board):
                task2_tictactoe(board)
                print("It's a draw :o")
                break
            
            ai_move = find_best_move(board, AI, YOU)
            board[ai_move] = AI
            if check_winner(board, AI):
                task2_tictactoe(board)
                print("AI wins :)")
                break
            elif is_board_full(board):
                task2_tictactoe(board)
                print("It's a draw :o")
                break
        else:
            print("That Spot is already Taken! Choose another Spot.")

main()