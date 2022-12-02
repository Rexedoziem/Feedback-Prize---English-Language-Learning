# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



import transformers
os.system('pip install iterative-stratification==0.1.7')
from transformers import AutoModel, AutoTokenizer, AutoConfig
import random
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.cuda import amp
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
import pickle
import gc
import tokenizers

class config:
    out_features = 6
    apex = True
    debug=False
    num_warmup_steps=0
    num_layers=3
    layer_start=4
    scheduler='cosine'
    learning_rate = 1e-5
    gradient_checkpointing=False
    DeBERTa_Model = "../input/deberta-v3-base"
    max_len = 1024
    num_cycles=0.5
    print_freq=20
    batch_scheduler=True
    EPOCHS = 4
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    train = True
    max_grad_norm=1000
    targets = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    BATCH_SIZE = 2
    seed=42
    learning_rate = 1e-5
    weight_decay = 0.01
    layerwise_learning_rate_decay = 0.9
    adam_epsilon = 1e-6
    encoder_lr=2e-5
    decoder_lr=2e-5
    num_warmup_steps = 0
    gradient_accumulation_steps = 1
    TRAINING_FILE = '/kaggle/input/feedback-prize-english-language-learning/train.csv'
if config.debug:
    config.epochs = 2
    config.trn_fold = [0]


tokenizer = AutoTokenizer.from_pretrained(config.DeBERTa_Model)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
config.tokenizer = tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_logger(filename=OUTPUT_DIR+'df'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)