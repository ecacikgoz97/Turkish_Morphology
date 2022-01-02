import argparse, torch, json, matplotlib, logging, os
from torch import optim
import matplotlib.pyplot as plt
from models.gpt3 import GPT3
from train import train
from common.utils import *
from data import build_data
matplotlib.use('Agg')

######## DON'T FORGET TO CHANGE THIS !!! ########
logger_file_name = 'deneme1'

# Loggers
if not os.path.exists('Transformer_Loggers'):
    os.mkdir('Transformer_Loggers')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')

logger_file_name = os.path.join('Transformer_Loggers', logger_file_name)
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.info('Code started \n')

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

# training info
args.batchsize = 128
args.epochs = 100
args.opt= 'WAdam'
args.lr = 0.001
args.task = 'lm'
args.seq_to_no_pad = 'surface'
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Using device: {}".format(str(args.device)))
logger.info("Training on '{}' is starting".format(args.seq_to_no_pad))
logger.info("We now give the hyperparameters")
logger.info("Number of Epochs : {}".format(args.epochs))
logger.info("Batch Size : {}".format(args.batchsize))
logger.info("Learning rate : {}".format(args.lr))


# data
#args.trndata = '/home/emrecan/Desktop/NLP/Turkish_Morphology/charLM/data/surf.uniquesurfs.trn.txt' # Linux
#args.valdata = '/home/emrecan/Desktop/NLP/Turkish_Morphology/charLM/data/surf.uniquesurfs.val.txt' # Linux
args.trndata = '/Users/emrecanacikgoz/Desktop/NLP/Turkish_Morphology/charLM/data/surf.uniquesurfs.trn.txt' # Mac
args.valdata = '/Users/emrecanacikgoz/Desktop/NLP/Turkish_Morphology/charLM/data/surf.uniquesurfs.val.txt' # Mac
args.tstdata = args.valdata
args.surface_vocab_file = args.trndata
#args.maxtrnsize = 57769; args.maxvalsize = 10000; args.maxtstsize = 10000
args.maxtrnsize = 5; args.maxvalsize = 5; args.maxtstsize = 5
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)

# model
args.mname = 'charlm_miniGPT'
num_layers=5
embed_dim=64
num_heads=16
block_size=128
embedding_dropout_rate=0.1 
attention_dropout_rate=0.1
residual_dropout_rate=0.1
expand_ratio = 4
args.model = GPT3(vocab=vocab,
                  num_layers=num_layers,
                  embed_dim=embed_dim,
                  num_heads=num_heads,
                  block_size=block_size,
                  embedding_dropout_rate=embedding_dropout_rate,
                  attention_dropout_rate=attention_dropout_rate,
                  residual_dropout_rate=residual_dropout_rate,
                  expand_ratio=expand_ratio)
args.model.to(args.device)

args.opt = optim.AdamW(args.model.parameters(), lr=args.lr, betas=(0.9, 0.95))
logger.info(f"> Number of parameters {len(torch.nn.utils.parameters_to_vector(args.model.parameters()))}")
logger.info(f"Number of Decoder Layers: {num_layers}")
logger.info(f"Embedding Dimension: {embed_dim}")
logger.info(f"Number of heads in Attention: {num_heads}")
logger.info(f"Block size (spatial extent of the model for its context): {block_size}")
logger.info(f"Embedding Dropout rate: {embedding_dropout_rate}")
logger.info(f"Attention Dropout rate: {attention_dropout_rate}")
logger.info(f"Residual Dropout rate: {residual_dropout_rate}")
logger.info(f"Expand ratio rate: {expand_ratio}")

# logging Results
modelname = args.mname+'/results/'+str(len(trndata))+'_instances'
args.results_file_name = os.path.join('Transformer_Loggers', modelname)
try:
    os.makedirs(args.results_file_name)
    print("Directory " , args.results_file_name,  " Created ")
except FileExistsError:
    print("Directory " , args.results_file_name,  " already exists")
args.save_path = args.results_file_name +  str(args.epochs)+'epochs.pt'
fig_path =  args.results_file_name +  str(args.epochs)+'epochs.png'
with open(args.results_file_name+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(vocab.word2id))

# plotting
args.fig, args.axs = plt.subplots(1)
args.plt_style = pstyle = '-'

# run
train(batches, logger, args)
plt.savefig(fig_path)
