import os
import sys
sys.path.append("../../")

from tqdm import tqdm
import numpy as np

import torch
from torch import nn,optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from IntentModel import IntentModel
from IntentDataset import IntentDataset
from config.IntentConfig import MODEL
from utils.evaluate import acc, evaluate

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Model
    model = IntentModel(MODEL)
    model.to(device)

    # Train & Val Datasets
    tr_ds = IntentDataset(True, dict_dir='../../train_tools/dict/chatbot_dict.json',
                          userDict_dir='../../utils/user_dic.tsv', maxlen=MODEL['maxlen'])
    tr_dl = DataLoader(dataset=tr_ds, 
                       batch_size=MODEL['batch_size'], 
                       shuffle=True, 
                       num_workers=4, 
                       drop_last=True)

    val_ds = IntentDataset(False, dict_dir='../../train_tools/dict/chatbot_dict.json', 
                           userDict_dir='../../utils/user_dic.tsv', maxlen=MODEL['maxlen'])
    val_dl = DataLoader(val_ds,
                        batch_size=MODEL['batch_size'],
                        shuffle=True,
                        num_workers=4,
                        drop_last=True)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # optim
    opt = optim.Adam(params=model.parameters(), lr=MODEL['learning_rate'])

    # schedular
    schedular = ReduceLROnPlateau(opt, factor=0.99)

    # train
    for epoch in tqdm(range(MODEL['epochs']), desc='epoch', total=MODEL['epochs']):
        schedular.step(epoch)
        tqdm.write("\nepoch : {}, lr : {}".format(epoch, opt.param_groups[0]['lr']))
        tr_loss = 0
        tr_acc = 0
        model.train()

        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            opt.zero_grad()
            
            query, intent = map(lambda elm: elm.to(device), mb)
            pred_intent = model(query)

            # loss 계산을 위해 shape 변경
            pred_intent = pred_intent.reshape(-1, pred_intent.size(-1))
            intent = intent.view(-1).long()

            mb_loss = loss_fn(pred_intent, intent.long()) # Input: (N, C) Target: (N)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(pred_intent, intent)

            tr_loss += mb_loss.item()
            tr_acc = mb_acc.item()
            tr_loss_avg =  tr_loss / (step + 1)
            tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}
            total_step = epoch * len(tr_dl) + step

        tqdm.write('\nepoch : {}, step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, 
                                                                                     total_step, 
                                                                                     tr_summary['loss'], 
                                                                                     tr_summary['acc']))
    
    model.eval()
    print("eval: ")
    val_summary = evaluate(model, loss_fn, val_dl, device)

    tqdm.write('\nval_loss: {: .3f}, val_acc{: .2%},'.format(val_summary['loss'],
                                                             val_summary['acc']))

    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.to(torch.device('cpu')).state_dict(),
        'opt_state_dict': opt.state_dict()
    }
            
    torch.save(state, 'best.tar')
            
if __name__ == '__main__':
    main()

