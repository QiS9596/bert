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
                if mode == 'concat4':
                    word_embedding = concat_embedding(bert_layers)
                elif mode == 'avg4':
                    # weights = [0.25,0.25,0.25,0.25]
                    num_layers = len(bert_layers)
                    weights = [1.0/(1.0*num_layers)]*num_layers
                    word_embedding = weighted_sum_embeddings(weights, bert_layers)
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

def weighted_sum_embeddings(weights, layers):
    """

    :param weights: list object for
    :param layers:
    :return: np.array which stores word embedding
    """
    layer_embedding = []
    for index in range(len(layers)):
        layer_value = np.array(layers[index]['values'])
        layer_value = layer_value* weights[index]
        layer_embedding.append(layer_value)
    layer_embedding = np.array(layer_embedding)
    result = np.sum(layer_embedding, axis=0)
    return result
# result = parse_bert_embedding_json('./tmp/vp_extract/label_feature.json')
# print(result.shape)
# print(result[0].shape)