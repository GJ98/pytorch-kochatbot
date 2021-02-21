from tqdm import tqdm

import torch

def evaluate(model, loss_fn, data_loader, device, ignore_idx=-1):
    if model.training:
        model.eval()

    summary = {'loss': 0, 'acc': 0}

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x, y = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            pred_y = model(x)

            # loss 계산을 위해 shape 변경
            pred_y = pred_y.reshape(-1, pred_y.size(-1))
            y = y.view(-1).long()

            # padding 제외한 value index 추출
            real_value_index = [y != ignore_idx]

            # acc 
            summary['acc'] += acc(pred_y[real_value_index], y[real_value_index])

            # loss
            summary['loss'] += loss_fn(pred_y, y.long()).item() #* dec_output.size()[0]

    # acc
    summary['acc'] /= step

    # loss
    summary['loss'] /= len(data_loader.dataset)

    return summary

def acc(yhat, y, ignore_idx=-1):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float().mean() # padding은 acc에서 제거
    return acc
