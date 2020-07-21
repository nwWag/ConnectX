# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
# See training.py for details. This file is needed to satisfy the requirements by kaggle to construct an agent as one Def.
# BUT IN CONTRAST TO TRAINING: we use not only the DQN as our agent but rather use the DQN only as the scoring function
# for a lookahead search. Significantly better.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def agent_function(observation, configuration):
    import torch
    from torch import nn
    import numpy as np
    import torch.nn.functional as F
    import numpy as np
    import random
    import math
    import base64
    import io
    import time
    from random import choice

    device = 'cpu'

    class Q_Network(nn.Module):
        def __init__(self, ngf = 64, actions=7, bn=False):
            super(Q_Network, self).__init__()
            self.ngf = ngf
            self.activation = nn.PReLU
            self.bn = bn

            self.conv = nn.Sequential(
                nn.Conv2d(3, ngf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3) if self.bn else nn.Identity(),
                self.activation(),

                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3) if self.bn else nn.Identity(),
                self.activation()
            )

            self.predict = nn.Sequential(nn.Linear(126, ngf), nn.BatchNorm1d(ngf) if self.bn else nn.Identity(), self.activation(),
                                        nn.Linear(ngf, ngf), nn.BatchNorm1d(ngf) if self.bn else nn.Identity(), self.activation(),
                                        nn.Linear(ngf, actions))#, nn.Softmax())

        def forward(self, x):
            rows = x.shape[2]
            columns = x.shape[3]
            x_new = torch.empty(x.shape[0], 3, rows, columns).to(device)
            for i in range(x.shape[0]):
                x_new[i] = F.one_hot(x[i].reshape(rows, columns).long(), 3).permute(2,0,1)

            x = x_new
            x = x.reshape(x.shape[0], -1).float()
            x = self.predict(x)

            return x
    
    
    def switch(board):
        board_is_list = type(board) is list
        if board_is_list:
            board = np.array(board)

        board[board == 1] = 3
        board[board == 2] = 1
        board[board == 3] = 2
        
        if board_is_list:
            board = list(board)
        return board

    def get_Qs(observation, configuration, module):
        rows = configuration['rows']
        columns = configuration['columns']
        input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, rows, columns)                
        Qs = module(input_observation)
        return Qs[0]

    def temp_agent(observation, configuration, module):
        if observation.mark != 1:
            observation['board'] = switch(observation['board'])
        return int(np.argmax(np.array([(q.item() if observation['board'][c] == 0 else -float('inf')) for c, q in enumerate(get_Qs(observation, configuration, module))])))

    # Negamax code from original connectx repo. ADAPTED to work with DQN and extended with alpha beta pruning.
    def nega_agent(obs, configuration, module):
        columns = configuration.columns
        rows = configuration.rows
        size = rows * columns

        def play(board, column, mark):
            columns = configuration.columns
            rows = configuration.rows
            row = max([r for r in range(rows) if board[column + (r * columns)] == 0])
            board[column + (row * columns)] = mark


        def is_win(board, column, mark, has_played=True):
            columns = configuration.columns
            rows = configuration.rows
            inarow = 4 - 1
            row = (
                min([r for r in range(rows) if board[column + (r * columns)] == mark])
                if has_played
                else max([r for r in range(rows) if board[column + (r * columns)] == 0])
            )

            def count(offset_row, offset_column):
                for i in range(1, inarow + 1):
                    r = row + offset_row * i
                    c = column + offset_column * i
                    if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or board[c + (r * columns)] != mark
                    ):
                        return i - 1
                return inarow

            return (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
            )

        # Due to compute/time constraints the tree depth must be limited.
        max_depth = 4

        def negamax_slow(board, mark, depth):
            # Can win next.
            for column in range(columns):
                if board[column] == 0 and is_win(board, column, mark, False):
                    return (torch.max(get_Qs({'board':board} if mark == 1 else {'board': switch(board)} , configuration, module)), column)

            # Recursively check all columns.
            best_score = -size
            best_column = None
            for column in range(columns):
                if board[column] == 0:
                    # Max depth reached. Score based on cell proximity for a clustering effect.
                    if depth <= 0:
                        score = torch.max(get_Qs({'board':board} if mark == 1 else {'board': switch(board)} , configuration, module))
                    else:
                        next_board = board[:]
                        play(next_board, column, mark)
                        (score, _) = negamax(next_board,
                                            1 if mark == 2 else 2, depth - 1)
                        score = score * -1

                    if score > best_score or (score == best_score and choice([True, False])):
                        best_score = score
                        best_column = column

            return (best_score, best_column)


        def negamax(board, mark, depth, alpha=-float('inf'), beta= float('inf')):
            # Can win next.
            for column in range(columns):
                if board[column] == 0 and is_win(board, column, mark, False):
                    return (torch.max(get_Qs({'board':board} if mark == 1 else {'board': switch(board)} , configuration, module)), column)

            # Recursively check all columns.
            best_score = -size
            best_column = None
            for column in range(columns):
                if board[column] == 0:
                    # Max depth reached. Score based on cell proximity for a clustering effect.
                    if depth <= 0:
                        score = torch.max(get_Qs({'board':board} if mark == 1 else {'board': switch(board)} , configuration, module))
                    else:
                        next_board = board[:]
                        play(next_board, column, mark)
                        (score, _) = negamax(next_board,
                                            1 if mark == 2 else 2, depth - 1,
                                            alpha=-beta, beta=-alpha)
                        score = score * -1

                    if score > best_score or (score == best_score and choice([True, False])):
                        best_score = score
                        best_column = column
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break

            return (best_score, best_column)

        _, column = negamax(obs['board'][:], obs['mark'], max_depth)
        if column == None:
            column = choice([c for c in range(columns) if obs['board'][c] == 0])
        return column


    module = Q_Network().to(device)
    encoded_weights = """
    BASE64_PARAMS"""
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    module.load_state_dict(torch.load(buffer, map_location=device))
    module.eval()

    return nega_agent(observation, configuration, module)