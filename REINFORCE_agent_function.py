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
        BASE64_PARAMS"""
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    module.load_state_dict(torch.load(buffer, map_location=device))
    return nega_agent(observation, configuration, module)


