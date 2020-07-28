import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, emb_dims, no_of_cont, emb_dropout=0.04):
        """
            Parameters
            ----------

            emb_dims: List of two element lists
              This list will contain a two element list for each
              categorical feature. The first element of a list will
              denote the number of unique values of the categorical
              feature. The second element will denote the embedding
              dimension to be used for that feature.

            no_of_cont: Integer
              The number of continuous features in the data.

            emb_dropout: Float
              The dropout to be used after the embedding layers.
            """

        super(Encoder, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        self.output_size = no_of_embs + no_of_cont

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def get_output_dim(self):

        return self.output_size

    def forward(self, cont_data, cat_data):
        """
        Forward propagation.
        """

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        return x # (batch_size, output_dim)


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim):
        """
        :param encoder_dim: feature size of encoded vectors
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.decoder_att = nn.Linear(decoder_dim, encoder_dim)  # linear layer to transform decoder's output
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded vectors, a tensor of dimension (batch_size, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = encoder_out  # (batch_size, encoder_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, encoder_dim)
        att = self.relu(att1 + att2)  # (batch_size, encoder_dim)
        alpha = self.softmax(att)  # (batch_size, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout=0.5):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: # unique pathways
        :param encoder_dim: feature size of encoder network
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded vectors, a tensor of dimension (batch_size, encoder_dim)
        :return: hidden state, cell state
        """
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_pathways, pathway_len):
        """
        Forward propagation.

        :param encoder_out: encoded vectors, a tensor of dimension (batch_size, encoder_dim)
        :param encoded_pathways: encoded pathways, a tensor of dimension (batch_size, max_caption_length)
        :param pathway_len: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for ward vocab, sorted encoded pathways, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        num_features = encoder_out.size(1)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths; why? apparent below
        pathway_len, sort_ind = pathway_len.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_pathways = encoded_pathways[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_pathways)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (pathway_len - 1).tolist()

        # Create tensors to hold ward prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_features).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_pathways, decode_lengths, alphas, sort_ind


class LoSDecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout=0.5):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: # unique pathways
        :param encoder_dim: feature size of encoder network
        :param dropout: dropout
        """
        super(LoSDecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, 1)  # linear layer to find scores over los
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded vectors, a tensor of dimension (batch_size, encoder_dim)
        :return: hidden state, cell state
        """
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_pathways, pathway_len, los):
        """
        Forward propagation.

        :param encoder_out: encoded vectors, a tensor of dimension (batch_size, encoder_dim)
        :param encoded_pathways: encoded pathways, a tensor of dimension (batch_size, max_caption_length)
        :param pathway_len: caption lengths, a tensor of dimension (batch_size, 1)
        :param los: LoS (batch_size, 1)
        :return: scores for ward vocab, sorted encoded pathways, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        num_features = encoder_out.size(1)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths; why? apparent below
        pathway_len, sort_ind = pathway_len.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_pathways = encoded_pathways[sort_ind]
        los = los[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_pathways)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (pathway_len - 1).tolist()

        # Create tensors to hold ward prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), 1).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_features).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, 1)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, los, decode_lengths, alphas, sort_ind
