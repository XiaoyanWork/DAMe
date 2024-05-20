"""
This file generates the initial message features

To leverage the semantics in the data, we generate document feature for each message,
which is calculated as an average of the pre-trained word embeddings of all the words in the message.
We use the word embeddings pre-trained by en_core_web_lg for English data, fr_core_web_lg for French data, and aravec (see https://github.com/bakrianoo/aravec) for Arabic data.

To leverage the temporal information in the data, we generate temporal feature for each message,
which is calculated by encoding the times-tamps: we convert each timestamp to OLE date,
whose fractional and integral components form a 2-d vector.

The initial feature of a message is the concatenation of its document feature and temporal feature.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import os
from enum import Enum

def bert_embedding(sentence):
    # 如果sentience是一个list类型
    if isinstance(sentence, list):
        sentence = " ".join(sentence)
    print(sentence)
    encoded_input = bert_tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: tensor.to('cuda') for key, tensor in encoded_input.items()}
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    model_output = mean_pooling(model_output, encoded_input['attention_mask']).squeeze()
    return model_output.detach().cpu().numpy()  # 不加跨语言樱色和

# encode messages to get features
def documents_to_features(df):
    # features = df.filtered_words.apply(bert_embedding).values
    features = df.text.apply(bert_embedding).values
    return np.stack(features, axis=0)



# encode one times-tamp
# t_str: a string of format '2012-10-11 07:19:34'
def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features

def get_sentence_model():
    pass

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_model():
    tokenizer = AutoTokenizer.from_pretrained('sentence_model')
    model = AutoModel.from_pretrained('sentence_model')
    return tokenizer, model

def main(dataset_dict): # dataset_dict path
    for key, value in dataset_dict.items():
        print("\nProcessing {}\n".format(key))
        t_features = df_to_t_features(value)
        d_features = documents_to_features(value)
        combined_features = np.concatenate((d_features, t_features), axis=1)
        np.save(save_path + 'features_filtered_{}.npy'.format(key), combined_features)


def get_datadict():
    dataset_dict = {}
    for dataset in dataset_lists:

        if dataset == "English_Twitter" or dataset == "Server_Twitter" or dataset == "English_Mix_Twitter":
            path = f"datasets/{dataset}/{dataset}.npy"
            df_np = np.load(path, allow_pickle=True)
            df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", \
                                                   "place_type", "place_full_name", "place_country_code", "hashtags",
                                                   "user_mentions", "image_urls", "entities",
                                                   "words", "filtered_words", "sampled_words"])
        elif dataset == "Arabic_Twitter" or dataset == "French_Twitter":
            path = f"datasets/{dataset}/{dataset}.npy"
            df_np = np.load(path, allow_pickle=True)
            df = pd.DataFrame(data=df_np, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
                                                   "hashtags", "urls", "words", "created_at", "filtered_words",
                                                   "entities",
                                                   "sampled_words"])
        elif dataset == "China_Twitter" or dataset == "Germany_Twitter" or dataset == "Japan_Twitter":
            path = f"datasets/{dataset}/{dataset}.json"
            df_json = pd.read_json(path)
            # 重命名
            df = df_json.rename(columns={"label": "event_id", "id": "tweet_id",
                                         "text": "text", "hashtags": "hashtags",
                                         "time": "created_at", "entities": "entities"})
            df['tweet_id'] = range(len(df))
            df['user_id'] = -1
            df['user_mentions'] = [[]] * len(df)

        else:
            raise NotImplementedError(f"不存在数据集{dataset}")

        dataset_dict[dataset] = df

    return dataset_dict

if __name__ == "__main__":
    # dataset_lists = ["Arabic_Twitter", "China_Twitter", "English_Twitter", "French_Twitter", "Germany_Twitter", "Japan_Twitter", "Server_Twitter", "English_Mix_Twitter"]
    dataset_lists = ["Arabic_Twitter", "China_Twitter", "English_Twitter", "French_Twitter", "Germany_Twitter", "Japan_Twitter"]
    # dataset_lists = ["English_Mix_Twitter"]
    save_path = "embeddings/"
    # 加载数据集
    dataset_dict = get_datadict()
    # 加载模型
    bert_tokenizer, bert_model = get_sentence_model()
    bert_model.cuda()

    main(dataset_dict)