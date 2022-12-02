import config
import transformers
import torch.nn as nn


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


# Mean Pooling for sentence representation 

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
  
 
class DebertaModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.DeBERTa_Model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.DeBERTa_Model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        self.pool = MeanPooling()
        self.pooler= WeightedLayerPooling(self.config.num_hidden_layers, config.layer_start, layer_weights=None)
        self.concat_pool = nn.Linear(self.config.hidden_size*4, self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, config.out_features)
        self._init_weights(self.fc)
        self._init_weights(self.concat_pool)
        
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_states = outputs.last_hidden_state # word level representation of last hiddent state
        # mean pooled sentence representation
        
        mean_feature = self.pool(last_hidden_states, attention_mask)
        
        all_hidden_states = torch.stack(outputs[1])
        weighted_pooling_embeddings = self.pooler(all_hidden_states)
        weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        
        
        # attention based sentence representation
        weights = self.attention(last_hidden_states).float()
        weights[attention_mask==0] = float('-inf')
        weights = torch.softmax(weights, 1)
        attention_feature = torch.sum(weights * last_hidden_states, dim=1)
        
        # CLS Token representation
        cls_token_feature = last_hidden_states[:, 0, :] # only cls token
        
        # Concat them
        combine_feature = torch.cat([mean_feature, attention_feature, cls_token_feature, weighted_pooling_embeddings], dim = -1)
        
        # MLP
        feature = self.concat_pool(combine_feature)
        return feature

    def forward(self, input_ids, attention_mask):
        feature = self.feature(input_ids, attention_mask)
        outputs = self.fc(feature)
        return outputs
    
    
model = DebertaModel(config, config_path=None, pretrained=True)
torch.save(model.config, OUTPUT_DIR+'config.pth')
model.to(device)