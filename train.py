from models import TextClassifierLSTM, TextClassifierTransformer, Visualizer, eval
from cfgs import get_total_settings
from data import DatasetTHUNews, collate_fn, collate_fn_trans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torchnet import meter
import os

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None: continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

if __name__ == '__main__':
    cfgs = get_total_settings()
    seed = cfgs.seed
    checkpoints_dir = cfgs.checkpoints_dir
    grad_clip = cfgs.grad_clip
    learning_rate = cfgs.learning_rate
    learning_rate_decay_start = cfgs.learning_rate_decay_start
    learning_rate_decay_every = cfgs.learning_rate_decay_every
    learning_rate_decay_rate = cfgs.learning_rate_decay_rate
    weight_decay = cfgs.weight_decay
    patience = cfgs.patience
    save_checkpoint_every = cfgs.save_checkpoint_every
    visualize_every = cfgs.visualize_every
    model_name = cfgs.model_name
    best_model_path = os.path.join(checkpoints_dir, model_name + '.pt')
    assert model_name in ['BiLSTM', 'Transformer'], 'Not implement yet'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    vis = Visualizer(env='train model')
    dataset = DatasetTHUNews(cfgs, mode='train')
    collate_fn = collate_fn if model_name == 'BiLSTM' else collate_fn_trans
    dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataset = DatasetTHUNews(cfgs, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfgs.batch_size, shuffle=True, collate_fn=collate_fn)
    cfgs.vocab_size = dataset.get_vocab_size()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifierLSTM(cfgs) if model_name == 'BiLSTM' else TextClassifierTransformer(cfgs)
    model = model.to(device)
    model.train()
    loss_function = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_meter = meter.AverageValueMeter()

    patience_cnt = 0
    epoch = 0
    best_score = None

    print('==================Training Begin==================')

    while(True):
        if patience_cnt >= patience: break
        loss_meter.reset()
        model.train()

        if learning_rate_decay_start != -1 and epoch > learning_rate_decay_start:
            frac = int((epoch - learning_rate_decay_start) // learning_rate_decay_every)
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = learning_rate * decay_factor
            set_learning_rate(optimizer, current_lr)

        for i, (sentences, label, length) in enumerate(dataloader):
            sentences = sentences.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            preds = model(sentences, length)
            loss = loss_function(preds, label)
            loss.backward()
            clip_gradient(optimizer, grad_clip)
            optimizer.step()
            train_loss = loss.detach()
            loss_meter.add(train_loss.item())
            # print('#############Over#############')

            if i % visualize_every == 0:
                vis.plot('train_loss', loss_meter.value()[0])
                information = 'best_score is ' + (str(best_score) if best_score is not None else '0.0')
                vis.log(information)

            is_best = False
            if (i + 1) % save_checkpoint_every == 0:
                model.eval()
                precision, recall, f_score = eval(cfgs, model, valid_dataloader, device)
                model.train()
                vis.log('{}'.format('=====F1 score is ' + str(f_score)) + ' iter= ' + str(i))

                if best_score is None or best_score < f_score:
                    is_best = True
                    best_score = f_score
                    patience_cnt = 1
                else:
                    patience_cnt += 1

                if is_best:
                    torch.save(model.state_dict(), best_model_path)
                # print('#############OK#############')

        epoch += 1

    print("==================Training End==================")

