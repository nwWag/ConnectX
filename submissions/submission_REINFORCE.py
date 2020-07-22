def agent_function(observation, configuration):
    import torch
    from torch import nn
    import numpy as np
    import torch.nn.functional as F
    import numpy as np
    import base64
    import io
    from torch import optim
    from torch.autograd import Variable
    from random import choice


    device = 'cpu'

    class PolicyNetwork(nn.Module):
        def __init__(self, num_inputs, num_actions, hidden_size=128, learning_rate=3e-4, gamma=0.90):
            super(PolicyNetwork, self).__init__()

            self.num_actions = num_actions
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.linear2 = nn.Linear(hidden_size, num_actions)
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            self.gamma = gamma

        def forward(self, state):
            x = F.relu(self.linear1(state))
            x = F.softmax(self.linear2(x), dim=1)
            return x

        def get_action(self, state):
            row = torch.tensor([state[c] for c in range(self.num_actions)])
            state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
            probs = self.forward(Variable(state))  # probs have shape (1 x num_actions)
            red_choices = torch.IntTensor([c for c in range(self.num_actions) if state.squeeze(0).numpy()[c] == 0])
            red_probs = probs.detach()[0, row == 0] / torch.sum(probs.detach()[0, row == 0])
            highest_prob_action = int(np.random.choice(red_choices, p=red_probs.numpy()))
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
            return -1 * log_prob


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


    def temp_agent(observation, configuration, module):
        if observation.mark != 1:
            observation['board'] = switch(observation['board'])
        action, _ = module.get_action(observation['board'])
        return action


    # Negamax code from original connectx repo. ADAPTED to work with REINFORCE and extended with alpha beta pruning.
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
                    return module.get_action(board if mark == 1 else switch(board)), column

            # Recursively check all columns.
            best_score = -size
            best_column = None
            for column in range(columns):
                if board[column] == 0:
                    # Max depth reached. Score based on cell proximity for a clustering effect.
                    if depth <= 0:
                        score = module.get_action(board if mark == 1 else switch(board))
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

        def negamax(board, mark, depth, alpha=-float('inf'), beta=float('inf')):
            # Can win next.
            for column in range(columns):
                if board[column] == 0 and is_win(board, column, mark, False):
                    return module.get_action(board if mark == 1 else switch(board)), column

            # Recursively check all columns.
            best_score = -size
            best_column = None
            for column in range(columns):
                if board[column] == 0:
                    # Max depth reached. Score based on cell proximity for a clustering effect.
                    if depth <= 0:
                        score = module.get_action(board if mark == 1 else switch(board))
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

    module = PolicyNetwork(num_inputs=42, num_actions=7).to(device)
    encoded_weights = """
        gAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAA
AGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAA
aW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAmNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxAClScQEo
WA4AAABsaW5lYXIxLndlaWdodHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQMo
KFgHAAAAc3RvcmFnZXEEY3RvcmNoCkZsb2F0U3RvcmFnZQpxBVgNAAAAMjc3NDU5MjQ3MDc2OHEG
WAMAAABjcHVxB00AFU50cQhRSwBLgEsqhnEJSypLAYZxColoAClScQt0cQxScQ1YDAAAAGxpbmVh
cjEuYmlhc3EOaAMoKGgEaAVYDQAAADI3NzQ1OTI0Njg5NDRxD2gHS4BOdHEQUUsAS4CFcRFLAYVx
EoloAClScRN0cRRScRVYDgAAAGxpbmVhcjIud2VpZ2h0cRZoAygoaARoBVgNAAAAMjc3NDU5MjQ2
NjM1MnEXaAdNgANOdHEYUUsASwdLgIZxGUuASwGGcRqJaAApUnEbdHEcUnEdWAwAAABsaW5lYXIy
LmJpYXNxHmgDKChoBGgFWA0AAAAyNzc0NTkyNDY2MTYwcR9oB0sHTnRxIFFLAEsHhXEhSwGFcSKJ
aAApUnEjdHEkUnEldX1xJlgJAAAAX21ldGFkYXRhcSdoAClScSgoWAAAAABxKX1xKlgHAAAAdmVy
c2lvbnErSwFzWAcAAABsaW5lYXIxcSx9cS1oK0sBc1gHAAAAbGluZWFyMnEufXEvaCtLAXN1c2Iu
gAJdcQAoWA0AAAAyNzc0NTkyNDY2MTYwcQFYDQAAADI3NzQ1OTI0NjYzNTJxAlgNAAAAMjc3NDU5
MjQ2ODk0NHEDWA0AAAAyNzc0NTkyNDcwNzY4cQRlLgcAAAAAAAAAIa2CPa6Ci720I4Y9wFQBPmDy
4T0kdZ09tNPQvYADAAAAAAAA8hBsPteQDD8Ts4O+ASfAPsb4bz4D5bK+RO0SvxLLwT7jj2M+QmVx
PmbNoD02hwi+o49EvT3BOT4irho9xJh0vjA3ar6sLl0+cJYTPi7XMT6jcAq+tsaYvoqqBj+vP6c+
+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
"""
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    module.load_state_dict(torch.load(buffer, map_location=device))
    return nega_agent(observation, configuration, module)


