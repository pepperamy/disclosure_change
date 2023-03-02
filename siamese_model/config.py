import torch
from torchinfo import summary
import random 
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pickle


class config:
    def __init__(self):
        self.emb_dim = 768
        self.wrd_len = 64  # 100
        self.para_len = 32  # 60
        self.num_filters = 128

        #attention
        self.key_size = 60
        self.value_size = 60
        self.query_size = 60

        #model
        self.kernel_sizes = (1, 10)
        self.kernel_sizes1 = (1,5)
        self.kernel_sizes2 = (5, 3)  # (2,2)
        self.kernel_sizes3 = (3, 3)
        self.dropout_rate = 0.2
        self.num_classes = 2.0
        self.num_labels = 2
        self.batch_size = 64
        self.class_weight = 1

        self.test_mode = False
        self.set_ct_loss = False
        self.model_path =  "/research/jujun/text_change/model/simple_siamese_"
        self.verbose_mode = True
        self.set_ct_loss = False
        self.imbalance = False


 

        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            id = 1
            torch.cuda.set_device(1)
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(id))
            print(torch.cuda.current_device())


    @staticmethod
    def test_model(model, model_config):
        # If there's a GPU available...
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            id = 1
            torch.cuda.set_device(1)
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(id))
            print(torch.cuda.current_device())
        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        summary(model, [(model_config.batch_size, model_config.para_len, model_config.wrd_len,
                768), (model_config.batch_size, model_config.para_len, model_config.wrd_len, 768)])


    
    def tokenize(self):
        #embedding
        print('Loading BERT tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def bert(self):
        self.bert_model = AutoModel.from_pretrained(
            # 'ProsusAI/finbert',
            'bert-base-uncased',
            # 'yiyanghkust/finbert-pretrain',
            num_labels = 2, 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )
        self.bert_model.cuda()


    def load_data(self):
           # load data
        self.para_map = pickle.load(open("/research/rliu/fraud/data/mda/paragraphs_1994_2016.pkl","rb"))
        pos_neg_pair = pd.read_csv('/research/jujun/text_change/data/pos_neg_pair.csv')
        pos_neg_pair = pos_neg_pair.dropna()

        self.imbalance =False
        if not self.imbalance:
            pos_index = pos_neg_pair[pos_neg_pair.fraud == 1].index[0:30]
            neg_index = pos_neg_pair[pos_neg_pair.fraud == 0].sample(len(pos_index)).index
            self.df = pos_neg_pair.loc[neg_index.append(pos_index),:]
            print(self.df.shape)
        else:
            pos_cik = list(set(pos_neg_pair[pos_neg_pair.fraud == 1].cik))
            neg_cik = list(set(pos_neg_pair[pos_neg_pair.fraud == 0].cik))
            neg_cik = [c for c in neg_cik if c not in pos_cik]
            neg_cik = random.sample(neg_cik, len(pos_cik))
            self.df = pos_neg_pair[pos_neg_pair.cik.isin(pos_cik[0:10] + neg_cik[0:10])]
            print(self.df.shape)
        print('successfully load data ...')
