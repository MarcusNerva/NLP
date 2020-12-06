from models import TextClassifierLSTM, TextClassifierTransformer, Visualizer, eval
from cfgs import get_total_settings
from data import DatasetTHUNews, collate_fn
import torch
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    cfgs = get_total_settings()
    seed = cfgs.seed
    checkpoints_dir = cfgs.checkpoints_dir
    checkpoints_path = os.path.join(checkpoints_dir, cfgs.model_name + '.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DatasetTHUNews(cfgs=cfgs, mode='test')
    cfgs.vocab_size = test_dataset.get_vocab_size()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, collate_fn=collate_fn)
    model = TextClassifierLSTM(cfgs) if cfgs.model_name == 'BiLSTM' else TextClassifierTransformer(cfgs)
    model.load_state_dict(torch.load(checkpoints_path))
    model.to(device)

    precision, recall, f_score = eval(cfgs, model, test_dataloader)
    print('###########precision == {precision}###########'.format(precision=precision))
    print('###########recall == {recall}###########'.format(recall=recall))
    print('###########f_score == {f_score}###########'.format(f_score=f_score))
