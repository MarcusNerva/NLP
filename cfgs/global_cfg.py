def get_total_settings():
    import argparse
    parser = argparse.ArgumentParser()

    """
    ========================Global Setting========================
    """
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checkpoints_dir', type=str, default='/home/hanhuaye/PythonProject/NLP/checkpoints')
    # parser.add_argument('--checkpoints_dir', type=str, default='/Users/bismarck/PycharmProjects/NLP/checkpoints')
    parser.add_argument('--model_name', type=str, default='BiLSTM', help='BiLSTM/Transformer')

    """
    ========================Data========================
    """
    parser.add_argument('--data_dir', type=str, default='/home/hanhuaye/PythonProject/NLP/data')
    # parser.add_argument('--data_dir', type=str, default='/Users/bismarck/PycharmProjects/NLP/data')
    parser.add_argument('--news_dir', type=str, default='/home/hanhuaye/PythonProject/NLP/data/THUCNews')
    # parser.add_argument('--news_dir', type=str, default='/Users/bismarck/PycharmProjects/NLP/data/fakenews')
    parser.add_argument('--word2int_path', type=str, default='/home/hanhuaye/PythonProject/NLP/data/word2int.pkl')
    # parser.add_argument('--word2int_path', type=str, default='/Users/bismarck/PycharmProjects/NLP/data/word2int.pkl')
    parser.add_argument('--sentences_path', type=str, default='/home/hanhuaye/PythonProject/NLP/data/digitized_sentences.pkl')
    # parser.add_argument('--sentences_path', type=str, default='/Users/bismarck/PycharmProjects/NLP/data/digitized_sentences.pkl')
    parser.add_argument('--train_path', type=str, default='/home/hanhuaye/PythonProject/NLP/data/train.pkl')
    # parser.add_argument('--train_path', type=str, default='/Users/bismarck/PycharmProjects/NLP/data/train.pkl')
    parser.add_argument('--valid_path', type=str, default='/home/hanhuaye/PythonProject/NLP/data/valid.pkl')
    # parser.add_argument('--valid_path', type=str, default='/Users/bismarck/PycharmProjects/NLP/data/valid.pkl')
    parser.add_argument('--test_path', type=str, default='/home/hanhuaye/PythonProject/NLP/data/test.pkl')
    # parser.add_argument('--test_path', type=str, default='/Users/bismarck/PycharmProjects/NLP/data/test.pkl')
    parser.add_argument('--text_numbers', type=int, default=-1)

    """
    ========================LSTM Settings========================
    """
    parser.add_argument('--word_size', type=int, default=512)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=-1)
    parser.add_argument('--is_bilstm', action='store_true', default=False)

    """
    ========================Transformer Settings========================
    """
    parser.add_argument('--src_pad_idx', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)


    """
    ========================classifier Settings========================
    """
    parser.add_argument('--class_number', type=int, default=14)


    """
    ========================Training Settings========================
    """
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--cut_length', type=int, default=50)

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=6, help='after how many iteration begin learning rate decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=4, help='for every x iteration learning rate have to decay')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    parser.add_argument('--visualize_every', type=int, default=10, help='show us loss every x iteration')
    parser.add_argument('--save_checkpoint_every', type=int, default=500)

    """
    ========================Testing Settings========================
    """
    parser.add_argument('--test_model', type=str, default='')


    """
    ========================Application Settings========================
    """
    parser.add_argument('--transformer_path', type=str, default='/home/hanhuaye/PythonProject/NLP/checkpoints/trans_2layers_40_3e-4.pt')
    parser.add_argument('--bilstm_path', type=str, default='/home/hanhuaye/PythonProject/NLP/checkpoints/BiLSTM_40_3e-4.pt')
    parser.add_argument('--hhy_word2int_path', type=str, default='/home/hanhuaye/PythonProject/NLP/data/word2int.pkl')


    args = parser.parse_args()
    return args