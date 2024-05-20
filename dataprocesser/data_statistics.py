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
from transformers import AutoTokenizer


def get_bert_token_len(value_list):
    count_list = []
    for sentience in value_list:
        # 如果sentience是一个list类型
        if isinstance(sentience, list):
            sentience = " ".join(sentience)
        print(sentience)
        encoded_input = bert_tokenizer(sentience, padding=True, truncation=True, return_tensors='pt')
        count_list.append(len(encoded_input["input_ids"].squeeze()))
    return np.array(count_list).mean()


def get_sentence_model():
    tokenizer = AutoTokenizer.from_pretrained('sentence_model')
    return tokenizer

def main(dataset_dict): # dataset_dict path
    data_dict = {}
    for key, value in dataset_dict.items():
        sub_dict = {}
        print("\nProcessing {}\n".format(key))
        sub_dict["Message_len"] = len(value)
        sub_dict["Events"] = len(value["event_id"].unique())
        sub_dict["token_avg_len"] = get_bert_token_len(value["text"])
        data_dict[key] = sub_dict
    np.save(save_path + "data_statistics.npy", data_dict)
    print(data_dict)

    for key, value in data_dict.items():
        print(key, "   " , value)

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
    dataset_lists = ["Arabic_Twitter", "China_Twitter", "English_Twitter", "French_Twitter", "Germany_Twitter", "Japan_Twitter", "Server_Twitter"]
    save_path = "embeddings/"
    # 加载数据集
    dataset_dict = get_datadict()
    # 加载模型
    bert_tokenizer = get_sentence_model()

    main(dataset_dict)