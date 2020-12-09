import torch
import torch.nn
from .text_classifier import TextClassifierTransformer, TextClassifierLSTM
import re
from zhon.hanzi import punctuation
import string
import pickle

total_punctuation = punctuation + '0123456789' + string.punctuation

class Application_Classifier:
    def __init__(self, cfgs):
        self.model_path = cfgs.hhy_transformer_path if cfgs.hhy_idx == 1 else cfgs.hhy_bilstm_path
        self.word2int_path = cfgs.hhy_word2int_path
        with open(self.word2int_path, 'rb') as f:
            self.word2int = pickle.load(f)
        cfgs.vocab_size = len(self.word2int) + 1

        self.model = TextClassifierTransformer(cfgs) if cfgs.hhy_idx == 1 else TextClassifierLSTM(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.news_type = ['constellation', 'entertainment', 'finance', 'home', 'lottery',
                          'politics', 'stock', 'education', 'fashion', 'game', 'house',
                          'pe', 'social', 'technology']
        self.total_punctuation = punctuation + '0123456789' + string.punctuation

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(device=self.device)

    def classify(self, text):
        text = re.sub('\s', '', text)
        text = re.sub(r'[%s]+' % self.total_punctuation, '', text)
        text = re.sub('[a-zA-Z]', '', text)
        numbers = []
        for i in range(len(text)):
            # if len(numbers) >= 40: break
            if text[i] not in self.word2int.keys(): continue

            numbers.append(self.word2int[text[i]])

        numbers = torch.LongTensor(numbers)
        numbers.unsqueeze(dim=0)
        out = self.model(numbers, len(numbers))
        out = torch.argmax(out, dim=1).item()
        return self.news_type[out]


