import config
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class DeBERTadataset:
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = config.TOKENIZER
        self.max_len = config.max_len
        self.texts = self.df['full_text']
        self.targets = self.df[config.targets].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = str(self.texts[index])
        texts = ' '.join(texts.split())


        inputs = self.tokenizer.encode_plus(
            texts, 
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            return_token_type_ids = True,
            return_attention_mask = True,
            truncation = True
        )


        resp = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long), 
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        }
        targets = torch.tensor(self.targets[index], dtype=torch.float)
        
        return resp, targets