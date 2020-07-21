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
            return highest_prob_action, log_prob


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


    module = PolicyNetwork(num_inputs=42, num_actions=7).to(device)
    encoded_weights = """
        BASE64_PARAMS"""
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    module.load_state_dict(torch.load(buffer, map_location=device))
    return temp_agent(observation, configuration, module)


