import config

df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv').fillna('none')

mskf= MultilabelStratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)

for fold, (trn_, val_) in enumerate(mskf.split(df, df[config.targets])):
    df.loc[val_, 'fold'] = int(fold)

df.to_csv('train_folds.csv', index=False)
df['fold'] = df['fold'].astype(int)
display(df.groupby('fold').size())

from tqdm import tqdm

# A quick check to get the number of text in the dataframe
lengths = []
tk0 = tqdm(df['full_text'].fillna("").values, total=len(df))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)
config.max_len = max(lengths) + 2 # cls & sep
LOGGER.info(f"max_len: {config.max_len}")