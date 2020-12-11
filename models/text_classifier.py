import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import pickle

def reverse_padded_sequence(inputs, lengths, batch_first=True):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


class Linearlayer(nn.Module):
    def __init__(self, seed, drop_prob, input_size, output_size, length):
        super(Linearlayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(length),
            nn.ReLU(True),
            nn.Dropout(drop_prob),
            nn.Linear(output_size, output_size)
        )

    def forward(self, features):
        # print(features.shape)
        return self.linear(features)


class TextClassifierLSTM(nn.Module):
    def __init__(self, args):
        super(TextClassifierLSTM, self).__init__()
        self.args = args
        self.word2int_path = args.hhy_word2int_path
        with open(self.word2int_path, 'rb') as f:
            self.word2int = pickle.load(f)
        args.vocab_size = len(self.word2int) + 1
        self.rnn_size = args.rnn_size
        self.word_size = args.word_size
        self.vocab_size = args.vocab_size
        self.class_number = args.class_number
        self.dropout = args.dropout
        self.seed = args.seed
        self.bilstm = args.is_bilstm
        self.lstm_num = 2 if self.bilstm else 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.news_type = ['constellation', 'entertainment', 'finance', 'home', 'lottery',
                          'politics', 'stock', 'education', 'fashion', 'game', 'house',
                          'pe', 'social', 'technology']

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.embed = nn.Embedding(self.vocab_size, self.rnn_size)
        self.lstm = nn.ModuleList()
        self.classifier = Linearlayer(seed=self.seed, drop_prob=self.dropout, input_size=self.rnn_size * self.lstm_num,
                                      output_size=self.class_number, length=self.class_number)
        # self.classifier = nn.Linear(in_features=self.rnn_size * self.lstm_num, out_features=self.class_number)
        self.softmax = nn.Softmax(dim=1)

        self.lstm.append(nn.LSTM(input_size=self.rnn_size, hidden_size=self.rnn_size, num_layers=1,
                                 batch_first=True, dropout=self.dropout, bidirectional=False))
        if self.bilstm:
            self.lstm.append(nn.LSTM(input_size=self.rnn_size, hidden_size=self.rnn_size, num_layers=1,
                                 batch_first=True, dropout=self.dropout, bidirectional=False))


    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.rnn_size).to(self.device),
                torch.zeros(1, batch_size, self.rnn_size).to(self.device))


    def forward(self, numberic, length):
        batch_size = numberic.shape[0]
        numberic = self.embed(numberic)
        hidden = [self.init_hidden(batch_size=batch_size) for l in range(self.lstm_num)]
        out = [numberic, reverse_padded_sequence(numberic, length, True)]

        for l in range(self.lstm_num):
            out[l] = rnn_utils.pack_padded_sequence(out[l], length, batch_first=True)
            out[l], hidden[l] = self.lstm[l](out[l], hidden[l])

            out[l], _ = rnn_utils.pad_packed_sequence(out[l], batch_first=True)
            if l == 1:
                out[l] = reverse_padded_sequence(out[l], length, batch_first=True)

        if self.lstm_num == 1:
            out = out[0]
        else:
            out = torch.cat(out, 2)

        out = torch.mean(out, dim=1)
        out = self.classifier(out)
        return out

    @torch.no_grad()
    def predict(self, text):
        import re
        from zhon.hanzi import punctuation
        import string

        self.load_state_dict(torch.load(self.args.bilstm_path))
        self.eval()
        self.to(self.device)

        total_punctuation = punctuation + '0123456789' + string.punctuation
        text = re.sub('\s', '', text)
        text = re.sub(r'[%s]+' % total_punctuation, '', text)
        text = re.sub('[a-zA-Z]', '', text)
        numbers = []
        for i in range(len(text)):
            if text[i] not in self.word2int.keys(): continue
            numbers.append(self.word2int[text[i]])

        numbers = torch.LongTensor(numbers)
        numbers = [numbers]
        numbers = rnn_utils.pad_sequence(numbers, batch_first=True, padding_value=0)
        numbers = numbers.to(self.device)

        out = self.forward(numbers, [len(numbers[0])])
        out = self.softmax(out)
        out = torch.argmax(out, dim=1).item()
        return self.news_type[out]


class TextClassifierTransformer(nn.Module):
    def __init__(self, cfgs):
        super(TextClassifierTransformer, self).__init__()
        self.cfgs = cfgs
        self.word2int_path = cfgs.hhy_word2int_path
        with open(self.word2int_path, 'rb') as f:
            self.word2int = pickle.load(f)
        cfgs.vocab_size = len(self.word2int) + 1
        self.vocab_size = cfgs.vocab_size
        self.src_pad_idx = cfgs.src_pad_idx
        self.d_model = cfgs.d_model
        self.dropout = cfgs.dropout
        self.class_number = cfgs.class_number
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = cfgs.seed
        self.n_heads = cfgs.n_heads
        self.num_layers = cfgs.num_layers
        self.news_type = ['constellation', 'entertainment', 'finance', 'home', 'lottery',
                          'politics', 'stock', 'education', 'fashion', 'game', 'house',
                          'pe', 'social', 'technology']

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.embed = nn.Embedding(self.vocab_size, self.d_model, self.src_pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.classifier = Linearlayer(seed=self.seed, drop_prob=self.dropout, input_size=self.d_model,
                                      output_size=self.class_number, length=self.class_number)
        # self.classifier = nn.Linear(in_features=self.d_model, out_features=self.class_number)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sentences, length):
        sentences = self.embed(sentences)
        output = self.transformer(sentences)
        output = torch.mean(output, dim=1)
        output = self.classifier(output)
        return output

    @torch.no_grad()
    def predict(self, text):
        import re
        from zhon.hanzi import punctuation
        import string

        self.load_state_dict(torch.load(self.cfgs.transformer_path))
        self.eval()
        self.to(self.device)

        total_punctuation = punctuation + '0123456789' + string.punctuation
        text = re.sub('\s', '', text)
        text = re.sub(r'[%s]+' % total_punctuation, '', text)
        text = re.sub('[a-zA-Z]', '', text)
        numbers = []
        for i in range(len(text)):
            if text[i] not in self.word2int.keys(): continue
            numbers.append(self.word2int[text[i]])

        numbers = torch.LongTensor(numbers)
        numbers = numbers.unsqueeze(dim=0)
        numbers = numbers.to(self.device)

        out = self.forward(numbers, None)
        out = self.softmax(out)
        out = torch.argmax(out, dim=1).item()
        return self.news_type[out]
