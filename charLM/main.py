import argparse, torch, json, matplotlib, os
import matplotlib.pyplot as plt
from models.charlm import CharLM
from common.utils import *
from data import build_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
from train import train

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'

# training
args.batchsize = 128
args.epochs = 35
args.opt= 'Adam'
args.lr = 0.001
args.task = 'lm'
args.seq_to_no_pad = 'surface'

# data
args.trndata = '/home/emrecan/Desktop/NLP/Turkish_Morphology/charLM/data/surf.uniquesurfs.trn.txt'
args.valdata = '/home/emrecan/Desktop/NLP/Turkish_Morphology/charLM/data/surf.uniquesurfs.val.txt'
args.tstdata = args.valdata
args.surface_vocab_file = args.trndata
args.maxtrnsize = 57769; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)

# model
args.mname = 'charlm'
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 64; args.nh = 350
args.enc_dropout_in = 0.2; args.enc_dropout_out = 0.3
args.model = CharLM(args, vocab, model_init, emb_init)
args.model.to(args.device)

# logging
args.modelname = 'logging/'+args.mname+'/results/'+str(len(trndata))+'_instances/'
try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ")
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
with open(args.modelname+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# plotting
args.fig, args.axs = plt.subplots(1)
args.plt_style = pstyle = '-'

# run
train(batches, args)
plt.savefig(args.fig_path)