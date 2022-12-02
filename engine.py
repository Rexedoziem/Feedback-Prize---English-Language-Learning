import config
import torch.nn as nn
from tqdm import tqdm
import torch

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")

def train_fx(fold, data_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.apex)
    train_loss = 0
    global_step = 0
    for step, (d, targets) in tqdm(enumerate(data_loader)):
        input_ids = d['input_ids']
        attention_mask = d['attention_mask']
        targets = targets
    # putting them into the device

        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        with torch.cuda.amp.autocast(enabled=config.apex):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
        #optimizer.zero_grad()
        
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        
        train_loss += loss.item() * d['input_ids'].size(0)
        train_loss = train_loss / len(data_loader)
    
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if config.batch_scheduler:
                scheduler.step()
        
        #if step % config.print_freq == 0 or step == (len(data_loader)-1):
    return train_loss

def eval_fx(data_loader, model, criterion, device):
    model.eval()
    valid_loss = 0

    #final_targets = []
    final_outputs = []
    with torch.no_grad():
        for step, (d, targets) in tqdm(enumerate(data_loader)):
            input_ids = d['input_ids']
            attention_mask = d['attention_mask']
            targets = targets
        
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            
            targets = targets.to(device, dtype=torch.float)
            # putting them into the device
            outputs = model(input_ids, attention_mask)
        
            loss = criterion(outputs, targets)
            

            valid_loss += loss.item() * d['input_ids'].size(0)
            valid_loss = valid_loss / len(data_loader)
            final_outputs.append(outputs.cpu().detach().numpy().tolist())
        
    final_outputs = np.concatenate(final_outputs)
    return valid_loss, final_outputs
  
torch.cuda.empty_cache()
gc.collect()

