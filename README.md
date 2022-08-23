# MTGNN-SUM
This repository contains the implementation for our paper: Multi Graph Neural Network for Extractive Long Document Summarization.

###Installation
The code is written in Python 3.6+. Its dependencies are summarized in the file requirements.txt. You can install these dependencies like this:
```shell
pip install -r requirements.txt
```

#### Datasets
Download Pubmed and Arxiv datasets from [here](https://github.com/armancohan/long-summarization)

#### Preprocess data
For pubmed dataset:
```shell
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task train
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task val
python preprocess_data.py --input_path dataset/pubmed-dataset --output_path dataset/pubmed --task test
```
For arxiv dataset:
```shell
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task train
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task val
python preprocess_data.py --input_path dataset/arxiv-dataset --output_path dataset/arxiv --task test
```

After getting the standard JSON format, you process the dataset by running a script: sh PrepareDataset.sh in the project directory. The processed files will be put under the cache directory.

#### Get contextualized embeddings

For pubmed dataset:
```shell
python feature_extraction.py --bert_model bert-base-uncased --data ./dataset/pubmed/train.label.jsonl --output ./bert_features_pubmed/bert_features_train --batch_size 100
python feature_extraction.py --bert_model bert-base-uncased --data ./dataset/pubmed/val.label.jsonl --output ./bert_features_pubmed/bert_features_val --batch_size 100
python feature_extraction.py --bert_model bert-base-uncased --data ./dataset/pubmed/test.label.jsonl --output ./bert_features_pubmed/bert_features_test --batch_size 100
```
For arxiv dataset:
```shell
python feature_extraction.py --bert_model bert-base-uncased --data ./dataset/arxiv/train.label.jsonl --output ./bert_features_arxiv/bert_features_train --batch_size 100
python feature_extraction.py --bert_model bert-base-uncased --data ./dataset/arxiv/val.label.jsonl --output ./bert_features_arxiv/bert_features_val --batch_size 100
python feature_extraction.py --bert_model bert-base-uncased --data ./dataset/arxiv/test.label.jsonl --output ./bert_features_arxiv/bert_features_test --batch_size 100
```
#### Training
Run command like this
```shell
python train.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path> --model [HSG|MTHSG] --save_root <model path> --log_root <log path> --bert_path <bert feature path> --lr_descent --grad_clip -m 3
```
For example:
```shell
python train.py --cuda --gpu 0 --data_dir dataset/arxiv --cache_dir cache/arxiv --embedding_path glove.42B.300d.txt --model MTHSG --save_root models_arxiv --log_root log_arxiv/ --bert_path bert_features_arxiv --lr_descent --grad_clip -m 3
```
#### Evaluation
For evaluation, the command may like this:
```shell
python evaluation.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path>  --model [HSG|HDSG] --save_root <model path> --log_root <log path> --bert_path <bert feature path> -m 5 --test_model multi --use_pyrouge
```
For example:
```shell
python evaluation.py --cuda --gpu 0 --data_dir dataset/arxiv --cache_dir cache/arxiv --embedding_path glove.42B.300d.txt  --model MTHSG --save_root models_arxiv --log_root log_arxiv/ --bert_path bert_features_arxiv -m 5 --test_model multi --use_pyrouge
```

**Note**: To use ROUGE evaluation, you need to download the 'ROUGE-1.5.5' package and then use pyrouge.

**Error Handling**:  If you encounter the error message Cannot open exception db file for reading: /path/to/ROUGE-1.5.5/data/WordNet-2.0.exc.db when using pyrouge, the problem can be solved from [here](https://github.com/tagucci/pythonrouge#error-handling).

Some code are borrowed from [HSG](https://github.com/dqwang122/HeterSumGraph). Thanks for their work.