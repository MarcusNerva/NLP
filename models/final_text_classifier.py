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
        self.model = TextClassifierTransformer(cfgs) if cfgs.hhy_idx == 1 else TextClassifierLSTM(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.news_type = ['constellation', 'entertainment', 'finance', 'home', 'lottery',
                          'politics', 'stock', 'education', 'fashion', 'game', 'house',
                          'pe', 'social', 'technology']
        self.total_punctuation = punctuation + '0123456789' + string.punctuation

        with open(self.word2int_path, 'rb') as f:
            self.word2int = pickle.load(f)
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

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from cfgs import get_total_settings
    args = get_total_settings()
    model = Application_Classifier(args)
    text = '北京时间12月9日凌晨，欧冠第6轮小组赛中，巴萨主场0-3负于尤文图斯，尤文获得头名出线，巴萨小组第2，欧冠主场不败金身告破。\
    C罗罚中自己创造的点球，欧冠生涯首次攻破巴萨球门，下半时点球梅开二度，近13场做客诺坎普斩获14球，\
    麦肯尼凌空抽射破门，连续两场建功，博努奇进球被判无效。'
    print(model.classify(text))
