import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup
import config
from sklearn import preprocessing
import engine
import re
import folds
import html
from sklearn import model_selection
import dataset
import torch
from model
from sklearn import metrics
import numpy as np

df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv').fillna('none')

def clean(text):
   # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\'', '', text)
    # sequences of white spaces
    #text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"^\s+|\s+$", "", text)
    
#text = re.sub(r'(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?', ' ', string)
    return text.lower()

full_text = df['full_text'].apply(clean)
df['full_text'] = full_text


def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[config.targets].values

    train_dataset = DeBERTadataset(
        train_folds, tokenizer = config.TOKENIZER,
        max_len = config.max_len)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = 2, pin_memory=True, drop_last=False)
    valid_dataset = DeBERTadataset(
        valid_folds,
        tokenizer = config.TOKENIZER,
        max_len = config.max_len
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.BATCH_SIZE * 2,
        shuffle = False,
        num_workers = 1, pin_memory=True, drop_last=False
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = DebertaModel(config, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_grouped_parameters(model, encoder_lr, decoder_lr, learning_rate, weight_decay, 
        layerwise_learning_rate_decay):
        
        no_decay = ["bias", 'LayerNorm.bias', "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "weight_decay": 0.0,
                "lr": decoder_lr,
            },
        ]
        #initialize lrs for every layer
        num_layers = model.config.num_hidden_layers
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = encoder_lr
        for layer in layers:
            lr *= layerwise_learning_rate_decay
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        return optimizer_grouped_parameters
    
    grouped_optimizer_params = get_optimizer_grouped_parameters(
    model, config.encoder_lr, config.decoder_lr, config.learning_rate, config.weight_decay, 
    config.layerwise_learning_rate_decay
    )
    
    optimizer = torch.optim.AdamW(
    grouped_optimizer_params,
    lr=config.learning_rate,
    eps=config.adam_epsilon
    )
    
    num_training_steps = int(len(train_folds) / config.BATCH_SIZE * config.EPOCHS)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / config.BATCH_SIZE * config.EPOCHS)
    scheduler = get_scheduler(config, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    
    best_score = np.inf

    for epoch in range(config.EPOCHS):

        # train
        train_loss = train_fx(fold, train_data_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        valid_loss, final_outputs = eval_fx(valid_data_loader, model, criterion, device)
        
        # scoring
        score, scores = get_score(valid_labels, final_outputs)
        

        LOGGER.info(f'Epoch {config.EPOCHS+1} - avg_train_loss: {train_loss:.4f}  avg_val_loss: {valid_loss:.4f}')
        LOGGER.info(f'Epoch {config.EPOCHS+1} - Score: {score:.4f}  Scores: {scores}')
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {config.EPOCHS+1} - Save Best Score: {best_score:.4f} Model')
            # CPMP: save the original model. It is stored as the module attribute of the DP model.
            torch.save({'model': model.state_dict(),
                        'final_outputs': final_outputs},
                        OUTPUT_DIR+f"{config.DeBERTa_Model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{config.DeBERTa_Model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['final_outputs']
    valid_folds[[f"pred_{c}" for c in config.targets]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds


if __name__ == '__main__':
    
    def get_result(oof_df):
        labels = oof_df[config.targets].values
        preds = oof_df[[f"pred_{c}" for c in config.targets]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

    if config.train:
        oof_df = pd.DataFrame()
        for fold in range(config.n_fold):
            if fold in config.trn_fold:
                _oof_df = train_loop(df, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')









