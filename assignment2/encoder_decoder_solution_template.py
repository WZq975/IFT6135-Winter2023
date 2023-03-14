import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(
            self,
            input_size, 
            hidden_size
            ):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)

    def forward(self, inputs, hidden_states=None):
        """GRU.

        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences. input_size corresponds to embedding size.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        batch_size, seq_len, _ = inputs.size()
        outputs = []
        if hidden_states is None:
            ht = torch.zeros((batch_size, self.hidden_size), device=next(self.parameters()).device)
        else:
            ht = hidden_states.squeeze(0)  # (batch_size, hidden_size)

        for t in range(seq_len):
            xt = inputs[:, t, :]  # (batch_size, input_size)

            rt = torch.sigmoid(torch.mm(xt, self.w_ir.t()) + self.b_ir + torch.mm(ht, self.w_hr.t()) + self.b_hr)
            zt = torch.sigmoid(torch.mm(xt, self.w_iz.t()) + self.b_iz + torch.mm(ht, self.w_hz.t()) + self.b_hz)
            nt = torch.tanh(torch.mm(xt, self.w_in.t()) + self.b_in + rt * (torch.mm(ht, self.w_hn.t()) + self.b_hn))
            ht = (1 - zt) * nt + zt * ht

            outputs.append(ht.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        ht = ht.unsqueeze(0)

        return outputs, ht


class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0 # note, this is an extrenous argument
        ):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size) # in the forwards, after multiplying
                                                     # do a torch.sum(..., keepdim=True), its a linear operation

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # Calculate the energy scores
        energy = self.tanh(self.W(torch.cat((inputs, hidden_states.permute(1, 0, 2).repeat(1, inputs.size(1), 1)), dim=2)))
        energy = torch.sum(self.V(energy), dim=2, keepdim=True)  # (batch_size, sequence_length, 1)

        if mask is not None:
            energy.masked_fill_(mask == 0, float('-inf'))

        x_attn = self.softmax(energy)  # (batch_size, sequence_length, 1)
        outputs = x_attn * inputs  # (batch_size, sequence_length, hidden_size)

        return outputs, x_attn


class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0
        ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            The input tensor containing the token sequences.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        embeddings = self.embedding(inputs)  # (batch_size, vocabulary_size, embedding_size)
        batch_size, seq_len, _ = embeddings.size()
        embeddings = self.dropout(embeddings)
        outputs, hidden_states = self.rnn(embeddings, hidden_states)
        # sum the two outputs from bidirectional rnns
        outputs = outputs[:, :, :self.rnn.hidden_size] + outputs[:, :, self.rnn.hidden_size:]
        # sum the two hidden_states from bidirectional rnns
        hidden_states = torch.sum(hidden_states, dim=0, keepdim=True)
        return outputs, hidden_states


    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0


class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0, 
        ):

        super(DecoderAttn, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=False, batch_first=True)
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        attn_outputs, attn_weights = self.mlp_attn(self.dropout(inputs), hidden_states, mask=mask)

        outputs, hidden_states = self.rnn(attn_outputs, hidden_states)

        return outputs, hidden_states
        
        
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False
        ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size,
                num_layers, dropout=dropout)
        if not encoder_only:
          self.decoder = DecoderAttn(vocabulary_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification. 
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states
