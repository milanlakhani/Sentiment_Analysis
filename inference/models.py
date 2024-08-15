import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob_1=0.5, drop_prob_2=0.3):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob_1, batch_first=True)
        self.dropout = nn.Dropout(drop_prob_2)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Stack up LSTM outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # Sigmoid function
        sig_out = self.sig(out)

        # Reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        return sig_out, hidden


    def init_hidden(self, batch_size, train_on_gpu=False):

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output, final_hidden_state):
        # attn_weights = self.attn(final_hidden_state)
        # attn_weights = torch.bmm(lstm_output, attn_weights.unsqueeze(2)).squeeze(2)
        # soft_attn_weights = self.softmax(attn_weights)
        # new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # return new_hidden_state
        # Expand the final_hidden_state to match lstm_output's sequence length
        final_hidden_state = final_hidden_state.unsqueeze(1).repeat(1, lstm_output.size(1), 1)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query=final_hidden_state, key=lstm_output, value=lstm_output)

        return attn_output.mean(dim=1)  # Average over the sequence length

class SentimentAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, num_heads=8, drop_prob_1=0.5, drop_prob_2=0.3):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob_1, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            dropout=drop_prob_1, batch_first=True)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.dropout = nn.Dropout(drop_prob_2)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        enc_lstm_out, hidden = self.encoder_lstm(embeds, hidden)

        # Pass the encoder output to the decoder
        dec_lstm_out, hidden = self.decoder_lstm(enc_lstm_out, hidden)

        # Apply attention mechanism
        attn_output = self.attention(dec_lstm_out, hidden[0][-1])

        # Dropout and fully-connected layer
        out = self.dropout(attn_output)
        out = self.fc(out)
        # Sigmoid function
        sig_out = self.sig(out)

        # Reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        return sig_out, hidden


    def init_hidden(self, batch_size, train_on_gpu = False):
        ''' Create two new tensors with sizes n_layers x batch_size x hidden_dim,
            initialized to zero, for hidden state and cell state of LSTM
        '''
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden