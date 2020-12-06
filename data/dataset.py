import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import os
import pickle
import sys
sys.path.append('../')
from cfgs import get_total_settings

class DatasetTHUNews(Dataset):
    """
    This dataset types comprises 3 kinds of mode, namely, train, valid, test.
    """

    def __init__(self, cfgs, mode):
        """
        Args:
            cfgs: the config of this project.
            mode: 'train', 'valid', 'test'
        """

        super(DatasetTHUNews, self).__init__()
        assert mode in ['train', 'valid', 'test'], '######ERROR:you have input a unimplemented mode######'
        dataset_path = cfgs.train_path
        word2int_path = cfgs.word2int_path
        if mode == 'valid':
            dataset_path = cfgs.valid_path
        elif mode == 'test':
            dataset_path = cfgs.test_path

        self.mode = mode
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        with open(word2int_path, 'rb') as f:
            self.word2int = pickle.load(f)


    def get_vocab_size(self):
        return len(self.word2int) + 1


    def __getitem__(self, idx):
        tuple = self.dataset[idx]
        numberic, label = tuple[0], tuple[1]
        numberic = np.array(numberic).astype(int)
        length = len(numberic)
        # print(type(numberic))

        return torch.tensor(numberic).long(), torch.tensor(label).long(), length

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    numberic, label, length = zip(*batch)

    numberic = rnn_utils.pad_sequence(numberic,batch_first=True, padding_value=0)
    label = torch.LongTensor(label)
    length = list(length)
    return numberic, label, length

if __name__=='__main__':
    from torch.utils.data import DataLoader

    cfgs = get_total_settings()
    dataset = DatasetTHUNews(mode='train', cfgs=cfgs)
    dataloader = DataLoader(dataset, cfgs.batch_size, shuffle=True, collate_fn=collate_fn)

    for i, (sentences, label, length) in enumerate(dataloader):
        print(sentences.shape)
        print(label)
        # print(length.shape)
        print(length)