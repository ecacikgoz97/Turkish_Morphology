import numpy as np
import torch

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
    epoch_wrong_predictions = [];
    epoch_correct_predictions = [];
    numbatches = len(batches)
    indices = list(range(numbatches))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx]
        loss, _acc, _ = args.model(surf)
        batch_loss = loss.sum()
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
    args.logger.write('%s --- avg_loss: %.4f, ppl: %.4f, acc: %.4f  \n' % (mode, nll,  ppl, acc))
    args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))
    f1 = open(args.modelname + "/"+str(args.epochs)+"epochs_"+ mode + "_wrong_predictions.txt", "w")
    f2 = open(args.modelname + "/"+str(args.epochs)+"epochs_"+ mode + "_correct_predictions.txt", "w")
    for i in epoch_wrong_predictions:
        f1.write(i+'\n')
    for i in epoch_correct_predictions:
        f2.write(i+'\n')
    f1.close(); f2.close()
    return nll, ppl, acc