import json
import numpy as np
def parse_bert_embedding_json(json_path, mode='concat4'):
    with open(json_path) as file:
        sentences = []
        for line in file.readlines():
            line_dict = json.loads(line)
            features = line_dict['features']
            sentence_embedding = []
            for word_token in features:
                bert_layers = word_token['layers']
                word_embedding = concat_embedding(bert_layers)
                sentence_embedding.append(word_embedding)
            sentences.append(np.array(sentence_embedding))
        sentences = np.array(sentences)
        return sentences


def concat_embedding(layers):
    layer_embedding = []
    for layer in layers:
        layer_embedding.append(layer['values'])
    result = np.concatenate(layer_embedding, axis=None)
    return result

# result = parse_bert_embedding_json('./tmp/vp_extract/label_feature.json')
# print(result.shape)
# print(result[0].shape)