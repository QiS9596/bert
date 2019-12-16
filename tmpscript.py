import parse_bert_embedding_json
import numpy as np
label_feature_path = './tmp/vp_extract/label_feature.json'
train_feature_path = './tmp/vp_extract/train_feature.json'
output_train_path = './data/vp/bert_embeddings/all_avg4.npy'
output_label_path = './data/vp/bert_embeddings/label_avg4.npy'
feature_train = parse_bert_embedding_json.parse_bert_embedding_json(json_path=train_feature_path,mode='avg4')
np.save(output_train_path, feature_train)
feature_label = parse_bert_embedding_json.parse_bert_embedding_json(json_path=label_feature_path, mode='avg4')
np.save(output_label_path, feature_label)


