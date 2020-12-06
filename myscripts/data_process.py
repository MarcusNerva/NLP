import re
import os
import glob
import sys
import pickle
import copy
from multiprocessing import Pool, cpu_count
from zhon.hanzi import punctuation
import string
from collections import defaultdict
sys.path.append('../')
from cfgs import get_total_settings

news_type = ['constellation', 'entertainment', 'finance', 'home', 'lottery', 'politics', 'stock',
'education', 'fashion', 'game', 'house', 'pe', 'social', 'technology']
total_punctuation = punctuation + '0123456789' + string.punctuation


def save_as_pkl(cfgs, news_idx, news_type):
    """
    This function aims to save processed sentences of each news type on disk respectively.
    We removed space, return and tab character from a given sentence.
    At the same time, statistics on different words are carried out.
    At last, sentences yet to digitize are saved on disk, and so do distinct words.
    Args:
        cfgs: the config of this project
        news_idx: [0, 14) index of a specific news type
        news_type: the types of all kinds of news

    Returns:
        None
    """
    data_dir = cfgs.data_dir
    news_dir = cfgs.news_dir
    news_name = news_type[news_idx]
    word_set = set()
    sentence_store = []

    specific_dir = os.path.join(news_dir, news_name)
    word_path = os.path.join(data_dir, news_name + '_wordsset.pkl')
    sentence_path = os.path.join(data_dir, news_name + '_sentences.pkl')

    txt_list = glob.glob(os.path.join(specific_dir, '*.txt'))
    txt_list = txt_list[:30000] if len(txt_list) > 30000 else txt_list
    for i in range(len(txt_list)):
        txt_path = txt_list[i]
        with open(txt_path, 'r', encoding='utf-8') as f:
            s = f.read()

        text = re.sub('\s', '', s)
        text = re.sub(r'[%s]+' % total_punctuation, '', text)
        text = re.sub('[a-zA-Z]', '', text)
        sentence_store.append(text)
        for j in range(len(text)):
            word_set.add(text[j])

    with open(word_path, 'wb') as f:
        pickle.dump(word_set, f)
    with open(sentence_path, 'wb') as f:
        pickle.dump(sentence_store, f)


def process_sentence(cfgs):
    """
    In this function, we trying to process sentences with multiprocess.
    Args:
        cfgs: the config of this project

    Returns:
        None
    """
    core_number = cpu_count()
    p = Pool(core_number)

    for i in range(len(news_type)):
        p.apply_async(save_as_pkl, args=(cfgs, i, news_type,))
    p.close()
    p.join()


def generate_word2int(cfgs):
    """
    Function 'generate_word2int' aim to generate word2int dictionary, and save this dictionary on disk.

    Args:
        cfgs: the config of this project.

    Returns:
        None.
    """
    data_dir = cfgs.data_dir
    word2int_path = os.path.join(data_dir, 'word2int.pkl')
    final_set = set()
    word2int = {}

    for i in range(len(news_type)):
        news_name = news_type[i]
        word_set_path = os.path.join(data_dir, news_name + '_wordsset.pkl')
        with open(word_set_path, 'rb') as f:
            word_set = pickle.load(f)
        final_set = final_set | word_set

    for idx, (word) in enumerate(final_set):
        word2int[word] = idx + 1

    with open(word2int_path, 'wb') as f:
        pickle.dump(word2int, f)


def sentence2numbers(cfgs, news_idx, news_type, word2int):
    """
    'sentence2numbers' aims to convert sentences_string into sentences_digit for NLP model convenience.

    Args:
        cfgs: the config of this project
        news_idx: [0, 14) index of a specific news type
        news_type: the types of all kinds of news
        word2int: dictionary for convert words into digits

    Returns:
        None
    """
    data_dir = cfgs.data_dir
    news_name = news_type[news_idx]
    sentences_path = os.path.join(data_dir, news_name + '_sentences.pkl')
    digitized_sentences_path = os.path.join(data_dir, news_name + '_digit_sentences.pkl')
    digitized_sentences = []

    with open(sentences_path, 'rb') as f:
        senteneces = pickle.load(f)
    for i in range(len(senteneces)):
        temp_sent = senteneces[i]
        crt_digit = []
        for j in range(len(temp_sent)):
            crt_digit.append(word2int[temp_sent[j]])
        digitized_sentences.append((crt_digit, news_idx))

    with open(digitized_sentences_path, 'wb') as f:
        pickle.dump(digitized_sentences, f)


def split_dataset(cfgs):
    """
    This function aims to split all the 14 kinds of news into train_dataset, valid_dataset, and test_dataset.
    These datasets are saved on the disk.
    In each kind of news, train_size: valid_size: test_size = 0.75 : 0.05 : 0.2.


    Args:
        cfgs: the config of this project.

    Returns:
        None
    """
    train_dataset, valid_dataset, test_dataset = [], [], []
    train_path, valid_path, test_path = cfgs.train_path, cfgs.valid_path, cfgs.test_path
    word2int_path = os.path.join(cfgs.data_dir, 'word2int.pkl')
    with open(word2int_path, 'rb') as f:
        word2int = pickle.load(f)

    core_number = cpu_count()
    p = Pool(core_number)
    for i in range(len(news_type)):
        p.apply_async(sentence2numbers, args=(cfgs, i, news_type, word2int, ))
    p.close()
    p.join()

    for i in range(len(news_type)):
        digitized_news_path = os.path.join(cfgs.data_dir, news_type[i] + '_digit_sentences.pkl')
        with open(digitized_news_path, 'rb') as f:
            digitized_news = pickle.load(f)
        total_len = len(digitized_news)
        train_len, valid_len = total_len * 0.75, total_len * 0.05

        for j in range(total_len):
            if j < train_len:
                train_dataset.append(digitized_news[j])
            elif j < train_len + valid_len:
                valid_dataset.append(digitized_news[j])
            else:
                test_dataset.append(digitized_news[j])


    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(valid_path, 'wb') as f:
        pickle.dump(valid_dataset, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)


if __name__ == '__main__':
    cfgs = get_total_settings()
    process_sentence(cfgs)
    generate_word2int(cfgs)
    split_dataset(cfgs)

    train_path = cfgs.train_path
    valid_path = cfgs.valid_path
    test_path = cfgs.test_path
    word2int_path = cfgs.word2int_path

    with open(train_path, 'rb') as f:
        train_set = pickle.load(f)
    with open(valid_path, 'rb') as f:
        valid_set = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_set = pickle.load(f)
    with open(word2int_path, 'rb') as f:
        word2int_dict = pickle.load(f)
    print('size of train_set is {siz}'.format(siz=len(train_set)))
    print('size of valid_set is {siz}'.format(siz=len(valid_set)))
    print('size of test_set is {siz}'.format(siz=len(test_set)))
    print('size of word2int is {siz}'.format(siz=len(word2int_dict)))

    for i, key in enumerate(word2int_dict):
        if i > 100: break
        print(key)
    # for i in range(10):
    #     print(len(train_set[i][0]))


