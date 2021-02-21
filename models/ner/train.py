import sys
sys.path.append('../../')

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from NerModel import NerModel
from NerDataset import NerDataset
from config.NerConfig import MODEL
from utils.evaluate import evaluate, acc

def main():
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    # Model
    model = NerModel(MODEL)
    model.to(device)

    # Train Datasets
    tr_ds = NerDataset(True, 
                       dict_dir='../../train_tools/dict/chatbot_dict.json',
                       maxlen=MODEL['maxlen'])

    tr_dl = DataLoader(dataset=tr_ds,
                       batch_size=MODEL['batch_size'],
                       shuffle=True,
                       num_workers=4,
                       drop_last=False)

    val_ds = NerDataset(False, 
                        dict_dir='../../train_tools/dict/chatbot_dict.json',
                        maxlen=MODEL['maxlen'])

    val_dl = DataLoader(dataset=val_ds,
                        batch_size=MODEL['batch_size'],
                        shuffle=True,
                        num_workers=4,
                        drop_last=False)

    # loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # optim
    opt = optim.Adam(params=model.parameters(),
                     lr=MODEL['learning_rate'])

    # schedular
    schedular = ReduceLROnPlateau(optimizer=opt,
                                  factor=0.99)

    # train
    for epoch in tqdm(range(MODEL['epochs']), desc='epoch', total=MODEL['epochs']):
        schedular.step(epoch)
        tqdm.write("\nepoch: {}, lr: {}".format(epoch, opt.param_groups[0]['lr']))
        tr_loss = 0
        tr_acc = 0
        model.train()

        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            opt.zero_grad()

            idx, tag = map(lambda elm: elm.to(device), mb)
            pred_tag = model(idx)

            # loss 계산을 위해 shape 변경
            pred_tag = pred_tag.reshape(-1, pred_tag.size(-1))
            tag = tag.view(-1).long()

            # padding 제외한 value index 추출
            real_value_index = [tag != 0]

            # padding은 loss 계산시 제외
            mb_loss = loss_fn(pred_tag[real_value_index], tag[real_value_index])
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(pred_tag[real_value_index], tag[real_value_index])
            
            tr_loss += mb_loss.item()
            tr_acc = mb_acc.item()
            tr_loss_avg = tr_loss / (step + 1)
            tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}
            total_step = epoch * len(tr_dl) + step

        tqdm.write('\nepoch : {}, step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, 
                                                                                     total_step, 
                                                                                     tr_summary['loss'], 
                                                                                     tr_summary['acc']))

    model.eval()
    print("eval: ")
    val_summary = evaluate(model, loss_fn, val_dl, device, ignore_idx=0)

    tqdm.write('\nval_loss: {: .3f}, val_acc{: .2%},'.format(val_summary['loss'],
                                                             val_summary['acc']))


    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.to(torch.device('cpu')).state_dict(),
        'opt_state_dict': opt.state_dict()
    }

    torch.save(state, 'best.tar')

if __name__=='__main__':
    main()




    