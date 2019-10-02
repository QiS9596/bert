import pandas as pd
import os
vp_data_path = './data/vp/'
base_file_name = 'vp16.base.word.shuffled.35.txt'
test_split=0.2
dev_split=0.1
df = pd.read_csv(os.path.join(vp_data_path,base_file_name), sep='\t', header=None, names=['labels','text'])
# print(df.sample(5))
# print(len(df.index)*test_split)
test_size=int(test_split*len(df.index))
dev_size=int(dev_split*len(df.index))
df = df.sample(frac=1.0)
test = df.iloc[0:test_size]
dev = df.iloc[test_size:test_size+dev_size]
train = df.iloc[test_size+dev_size:]
test.to_csv(os.path.join(vp_data_path,'test.tsv'),header=False, sep='\t',index=False)
dev.to_csv(os.path.join(vp_data_path,'dev.tsv'),header=False, sep='\t',index=False)
train.to_csv(os.path.join(vp_data_path,'train.tsv'),header=False, sep='\t',index=False)
print(len(df.index))
print(len(train.index))
print(len(dev.index))
print(len(test.index))