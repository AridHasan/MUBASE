# MUBASE - MUltiplatform BAngla SEntiment

The MUBASE dataset is a multiplatform dataset consisting of Tweets and Facebook posts, which are manually annotated with sentiment polarity.
The annotation agreement of this manually annotated dataset shows an agreement score of 0.84, indicating a perfect agreement among the annotators.

## Dataset

 - [training set](./Dataset/MUBASE_train.tsv)
 - [development set](./Dataset/MUBASE_dev.tsv)
 - [test set](./Dataset/MUBASE_test.tsv)

## [Code](./code)

Code folder includes all the scripts that we used for our experiments. To run the scripts, please follow the bellow commands:

#### Classical Algorithm

To run SVM:
```
python code/classical_model.py -a svm -i ./Dataset/MUBASE_train.tsv -v ./Dataset/MUBASE_dev.tsv -t ./Dataset/MUBASE_test.tsv -o svm.txt -m model/
```
To run RF:
```
python code/classical_model.py -a rf -i ./Dataset/MUBASE_train.tsv -v ./Dataset/MUBASE_dev.tsv -t ./Dataset/MUBASE_test.tsv -o svm.txt -m model/
```

#### Fine-tuned Feed Forward Net with Embeddings
- [Dev set Link](https://drive.google.com/file/d/1qpTj5coV04DLayvCJzoInUL8uIjsHhWp/view?usp=sharing)
- [Train set Link](https://drive.google.com/file/d/1vqa-V2twUYHiUwLgOSo1xT5H2_mbVvb-/view?usp=sharing)
- [Test set Link](https://drive.google.com/file/d/1_qyI1sbCMoChDBWtwAvcspi7E1NDH4_o/view?usp=sharing)

To run Fine-tuned FF:
```
python code/ffnet_pytorch_lightning.py --train data/train_ada_embeddings.jsonl --dev data/dev_ada_embeddings.jsonl --test data/test_embeddings.jsonl --model-dir ./model/ --results-dir em_output/
```

#### Fine-Tuned LLMs
To run BERT-m:

```
python code/run_glue_v1.py \
  --model_name_or_path bert-base-multilingual-uncased \
  --train_file ./Dataset/MUBASE_train.csv \
  --validation_file ./Dataset/MUBASE_dev.csv \
  --test_file ./Dataset/MUBASE_test.csv \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./bert-base-multilingual-uncased/ \
  --do_eval \
  --do_train \
  --overwrite_output_dir
```

To run BERT-bn:

```
python code/run_glue_v1.py \
  --model_name_or_path csebuetnlp/banglabert \
  --train_file ./Dataset/MUBASE_train.csv \
  --validation_file ./Dataset/MUBASE_dev.csv \
  --test_file ./Dataset/MUBASE_test.csv \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./banglabert/ \
  --do_eval \
  --do_train \
  --overwrite_output_dir
```

To run XLM-RoBERTa-base:

```
python code/run_glue_v1.py \
  --model_name_or_path xlm-roberta-base \
  --train_file ./Dataset/MUBASE_train.csv \
  --validation_file ./Dataset/MUBASE_dev.csv \
  --test_file ./Dataset/MUBASE_test.csv \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./xlm-roberta-base/ \
  --do_eval \
  --do_train \
  --overwrite_output_dir
```

To run XLM-RoBERTa-large:

```
python code/run_glue_v1.py \
  --model_name_or_path xlm-roberta-large \
  --train_file ./Dataset/MUBASE_train.csv \
  --validation_file ./Dataset/MUBASE_dev.csv \
  --test_file ./Dataset/MUBASE_test.csv \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./xlm-roberta-large/ \
  --do_eval \
  --do_train \
  --overwrite_output_dir
```
#### Bloomz

To run Bloomz model, it requires to host in a server.

## Citation

```
@article{hasan2023zero,
  title={Zero-and Few-Shot Prompting with LLMs: A Comparative Study with Fine-tuned Models for Bangla Sentiment Analysis},
  author={Hasan, Md Arid and Das, Shudipta and Anjum, Afiyat and Alam, Firoj and Anjum, Anika and Sarker, Avijit and Noori, Sheak Rashed Haider},
  journal={arXiv preprint arXiv:2308.10783},
  year={2023}
}
```


## License
MUBASE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this work. If not, see [http://creativecommons.org/licenses/by-nc-sa/4.0/](http://creativecommons.org/licenses/by-nc-sa/4.0/).