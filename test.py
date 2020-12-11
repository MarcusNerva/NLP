from models import TextClassifierLSTM, TextClassifierTransformer, Visualizer, eval
from cfgs import get_total_settings
from data import DatasetTHUNews, collate_fn, collate_fn_trans
import torch
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    cfgs = get_total_settings()
    seed = cfgs.seed
    test_model = cfgs.test_model
    model_name = cfgs.model_name
    checkpoints_dir = cfgs.checkpoints_dir
    checkpoints_path = os.path.join(checkpoints_dir, test_model + '.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert model_name in ['BiLSTM', 'Transformer'], 'Not Implemented! '

    collate_fn = collate_fn if model_name == 'BiLSTM' else collate_fn_trans
    test_dataset = DatasetTHUNews(cfgs=cfgs, mode='test')
    cfgs.vocab_size = test_dataset.get_vocab_size()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, collate_fn=collate_fn)
    model_name = test_model.split('_')[0]
    model = TextClassifierLSTM(cfgs) if model_name == 'BiLSTM' else TextClassifierTransformer(cfgs)
    model.load_state_dict(torch.load(checkpoints_path))
    model.to(device)

    model.eval()
    precision, recall, f_score, matrix = eval(cfgs, model, test_dataloader, device, is_test=True)
    print('###########precision == {precision}###########'.format(precision=precision))
    print('###########recall == {recall}###########'.format(recall=recall))
    print('###########f_score == {f_score}###########'.format(f_score=f_score))
    matrix_path = os.path.join(cfgs.checkpoints_dir, model_name + '_matrix.npy')
    import numpy as np
    np.save(matrix_path, matrix)
