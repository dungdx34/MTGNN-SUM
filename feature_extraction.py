'''
Dung Doan
'''

import argparse
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import numpy as np
import json

class InputFeatures(object):
    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask

def convert_examples_to_sentence_features(example, max_seq_length, tokenizer):
    feature = []
    for sent in example:

        tokens = tokenizer.encode(sent)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        feature.append(InputFeatures(input_ids=input_ids, input_mask=input_mask))

    return feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='./BERT/uncased_L-12_H-768_A-12', type=str)

    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--cache_dir", default="./bert_cache", type=str)

    parser.add_argument("--data", type=str, default='./dataset/pubmed/train.label.jsonl')

    parser.add_argument("--align_matrix", type=str)
    parser.add_argument("--layers", type=str, default='9:13')
    parser.add_argument("--batch_size", default=100, type=int)

    parser.add_argument("--output", type=str, default='./bert_features_pubmed/bert_features_train')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    max_seq_len = 100

    model = BertModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir, output_hidden_states=True)
    model.to(device)

    with open(args.data, encoding="utf-8") as f:
        line = f.readline()
        count = 0
        while line:
            document = json.loads(line)
            example = document["text"]

            features = convert_examples_to_sentence_features(example, max_seq_len, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

            train_data = TensorDataset(all_input_ids, all_input_mask)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

            layer_1 = int(args.layers.split(':')[0])
            layer_2 = int(args.layers.split(':')[1])

            if args.align_matrix:
                W = []
                for i in range(layer_1, layer_2):
                    temp = np.loadtxt(args.align_matrix + '.' + str(i) + '.map')
                    temp = torch.tensor(temp, dtype=torch.float).to(device)
                    W.append(temp)

            model.eval()
            to_save = {}
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask = batch

                with torch.no_grad():
                    _, _, all_encoder_layers = model(input_ids, attention_mask=input_mask)

                output = []
                for i, j in enumerate(range(layer_1, layer_2)):
                    if args.align_matrix:
                        output.append(torch.matmul(all_encoder_layers[j], W[i]))
                    else:
                        output.append(all_encoder_layers[j])
                output_ = torch.sum(torch.stack(output), dim=0)

                for i in range(len(input_ids)):
                    sent_id = i + step * args.batch_size
                    layer_output = output_[i, :input_mask[i].to('cpu').sum()]
                    sent = layer_output.detach().cpu().numpy()
                    to_save[sent_id] = sent[0]

            torch.save(to_save, args.output + "_doc_" +  str(count) + '.pth')
            line = f.readline()
            count += 1

if __name__ == "__main__":
    main()