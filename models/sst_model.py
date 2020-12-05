import pdb
import torch
import torch.nn as nn

class SST(nn.Module):
    """
    Container module with 1D convolutions to generate proposals
    """

    def __init__(self, opt):
        super(SST, self).__init__()
        self.scores = torch.nn.Linear(opt.hidden_dim, opt.K)

        # Saving arguments
        self.video_dim = opt.video_dim
        #self.W = opt.W
        self.rnn_type = opt.tap_rnn_type
        self.rnn_num_layers = opt.rnn_num_layers
        self.rnn_dropout = opt.rnn_dropout
        self.K = opt.K
        self.data_for_test = []
        self.rnn = nn.LSTM(opt.video_dim, opt.hidden_dim, opt.rnn_num_layers, batch_first=True,
                               dropout=opt.rnn_dropout)

    def eval(self):
        self.rnn.dropout = 0

    def train(self):
        self.rnn.dropout = self.rnn_dropout

    def forward(self, features):
        if hasattr(self, 'reduce_dim_layer'):
            features = self.reduce_dim_layer(features)
        features = features.unsqueeze(0)
        N, T, _ = features.size()
        rnn_output, _ = self.rnn(features)
        rnn_output = rnn_output.contiguous()
        rnn_output = rnn_output.view(rnn_output.size(0) * rnn_output.size(1), rnn_output.size(2))
        outputs = torch.sigmoid(self.scores(rnn_output)).view(N, T, self.K)
        return rnn_output, outputs.squeeze(0)