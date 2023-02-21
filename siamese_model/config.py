import torch
from torchinfo import summary
import random 
import pandas as pd
import numpy as np

    def __init__(self):
        self.emb_dim = 768
        self.wrd_len = 64  # 100
        self.para_len = 32  # 60
        self.num_filters = 128
        self.kernel_sizes = (1, 10)
        self.kernel_sizes2 = (5, 3)  # (2,2)
        self.kernel_sizes3 = (3, 3)
        self.dropout_rate = 0.2
        self.num_classes = 2.0
        self.num_labels = 2
        self.batch_size = 64
        self.para_map = None
        self.class_weight = 1
        self.test_mode = False
        self.set_ct_loss = False
        self.model_path =  "./model/simple_siamese_"
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

        #embedding
        print('Loading BERT tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


        self.bert_model = AutoModel.from_pretrained(
            # 'ProsusAI/finbert',
            'bert-base-uncased',
            # 'yiyanghkust/finbert-pretrain',
            num_labels = 2, 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )
        self.bert_model.cuda()


    def set_parm_map(self, para_map):
        self.para_map = para_map

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

    @staticmethod
    def set_seed():
        seed_val = 1234
        verbose_mode = True
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)