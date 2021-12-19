import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from models.gpt3 import GPT3
import numpy as np
import random
from test import test
from common.utils import *

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    #opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    opt = torch.optim.AdamW(args.model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5)
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4; best_ppl = 0
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
        epoch_wrong_predictions = [];
        epoch_correct_predictions = [];
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx]

            loss, _acc, _ = args.model(surf)
            batch_loss = loss.sum() #mean(dim=-1)
            batch_loss.backward()
            opt.step()
            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = _acc
            epoch_num_tokens += num_tokens
            epoch_loss       += batch_loss.item()
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions += wrong_predictions
            epoch_correct_predictions += correct_predictions
        nll = epoch_loss / numbatches
        ppl = np.exp(epoch_loss / epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens
        error = epoch_error / epoch_num_tokens
        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, ppl: %.4f, acc: %.4f \n' % (epc, nll,  ppl, acc))
        args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))
        f1 = open(args.modelname + "/"+str(args.epochs)+"epochs_trn_wrong_predictions.txt", "w")
        f2 = open(args.modelname + "/"+str(args.epochs)+"epochs_trn_correct_predictions.txt", "w")
        for i in epoch_wrong_predictions:
            f1.write(i+'\n')
        for i in epoch_correct_predictions:
            f2.write(i+'\n')
        f1.close(); f2.close()
        # VAL
        args.model.eval()
        with torch.no_grad():
            nll, ppl, acc = test(valbatches, "val", args)
            loss = nll
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        #scheduler.step(nll)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            best_ppl = ppl
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()
    plot_curves(args.task, args.model, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.model, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')