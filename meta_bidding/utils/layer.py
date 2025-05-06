import math
import torch
from torch import nn, Tensor

class FlatTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 2:
            return self.sigmoid(x[:,0])*self.tanh(x[:,1])
        elif len(x.shape) == 3:
            return self.sigmoid(x[:,:,0])*self.tanh(x[:,:,1])
        else:
            raise NotImplementedError



    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.swapaxes(0,1) # convert to batch first
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe # The length is fixed
        return self.dropout(x)
    
class Embedding(nn.Identity):
    """
        This embedding is defined, but does not change the input.
    """
    pass

class TransformerModel(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        num_tokens, # vocab size(which should be 1 for continous output, and action_dim for discrete output)
        d_model, # embedding size of each input token
        nhead, # number of heads in the multi-head attention layer
        num_encoder_layers, # number of encoder layers
        num_decoder_layers, # number of decoder layers
        dim_feedforward, # dimension of the feedforward network model
        dropout, # dropout probability
        encoder_token_dim, # dimension of the encoder token
        decoder_token_dim, # dimension of the decoder token
        param_token_dim, # dimension of the parameter tokens
        position_encoding_len = 24,
        device = "cuda"
    ):
        super(TransformerModel, self).__init__()
 

        # INFO
        self.model_type = "Transformer"
        self.d_model = d_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=position_encoding_len # Maximun 24 hours
        )
        self.encoder_embedding = nn.Sequential(nn.Linear(encoder_token_dim, d_model),nn.ReLU())
        self.decoder_embedding = nn.Sequential(nn.Linear(decoder_token_dim, d_model),nn.ReLU())
        self.param_embedding = nn.Sequential(nn.Linear(param_token_dim, d_model),nn.ReLU())
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.out = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Linear(dim_feedforward,dim_feedforward),
                                        nn.ReLU(),
                                        nn.Linear(dim_feedforward,1),
                                        nn.Tanh()).to(device)
        self.device = device
        self.to(device)

    def forward(
        self,
        src,
        tgt,
        param,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # # Embedding + positional encoding - Out size = (batch_size, sequence length, d_model)
        # batch_size,_,encoder_token_dim = src.size()
        # src_embedding = self.positional_encoder.generate(batch_size,24)
        # src = self.encoder_embedding(src)
        # src = torch.cat([src, src_embedding],dim=-1)

        # tgt_embedding = self.positional_encoder.generate(batch_size,24)
        # tgt = self.decoder_embedding(tgt)
        # tgt = torch.cat([tgt, tgt_embedding],dim=-1)

        # Typical Transformer encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # parameters encoding
        param = self.param_embedding(param).unsqueeze(-2) # (batch_size, 1, d_model)
        src = torch.cat([src, param],dim=-2) # (batch_size, 24+1, d_model)

        # Create target mask for decoder; (sz,sz), with dignal zeros and others -torch.inf
        sz = tgt.size(1)
        tgt_mask = torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1) + \
                   torch.tril(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=-1)

        # we permute to obtain size (sequence length, batch_size, d_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt,tgt_mask = tgt_mask)
        out = self.out(transformer_out)

        return out
    



class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNLSTMModel, self).__init__()

        # Conv1D layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Dense layer
        self.fc = nn.Linear(8, 256)

        # Bi-directional LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, dropout=0.1, bidirectional=True)

    def forward(self, x):
        # input shape: (timestep, channel, dim)

        # Conv1D layers
        x = self.conv1(x)

        # Flatten the last 2 dimensions of x, and apply dense layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Unflatten for LSTM
        x = x.view(x.size(0), -1, 256)

        # Bi-directional LSTM
        x, _ = self.lstm(x)

        # Only return the output of the last timestep
        x = x[-1]

        return x

class ClassWiseLinear(nn.Module):
    def __init__(self,num_class, in_features, out_features, bias=True):
        super().__init__()
        self.num_class = num_class
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.Tensor(num_class, in_features, out_features)
        if bias:
            self.biases = torch.Tensor(num_class, 1, out_features)
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        for w in self.weights:
            w.transpose_(0, 1)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            w.transpose_(0, 1)

        self.weights = nn.Parameter(self.weights)

        if self.biases is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[0].T)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.biases, -bound, bound)
            self.biases = nn.Parameter(self.biases)

    def forward(self, input):
        # input shape: (batch_size, num_class, dim_feature)
        return torch.baddbmm(self.biases, input.swapaxes(0,1), self.weights).swapaxes(0,1)

    def extra_repr(self) -> str:
        return 'num_class = {}, in_features={}, out_features={}, biases={}'.format(
            self.num_class, self.in_features, self.out_features, self.biases is not None
        )