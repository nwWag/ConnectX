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

    device = 'cpu'

    class Q_Network(nn.Module):
        def __init__(self, ngf = 64, actions=7, bn=False):
            super(Q_Network, self).__init__()
            self.ngf = ngf
            self.activation = nn.ReLU
            self.bn = bn

            self.conv = nn.Sequential(
                nn.Conv2d(3, ngf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3) if self.bn else nn.Identity(),
                self.activation(),

                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
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
        board[board == 1] = 3
        board[board == 2] = 1
        board[board == 3] = 2
        return board

    def get_Qs(observation, configuration, module):
        rows = configuration['rows']
        columns = configuration['columns']
        input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, rows, columns)                
        Qs = module(input_observation)
        return Qs[0]

    def temp_agent(observation, configurationm, module):
        if observation.mark != 1:
            observation = switch(observation)
        return int(np.argmax(np.array([(q.item() if observation['board'][c] == 0 else -float('inf')) for c, q in enumerate(get_Qs(observation, configuration, module))])))


    module = Q_Network()
    encoded_weights = """
    BASE64_PARAMS"""
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    module.load_state_dict(torch.load(buffer, map_location=device))
    module.eval()
    return temp_agent(observation, configuration, module)