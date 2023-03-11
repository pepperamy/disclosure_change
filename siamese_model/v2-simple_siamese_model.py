

import torch.nn as nn
import torch
import math
from torchmetrics.functional import pairwise_cosine_similarity

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def masked_softmax(self, config, scores, mask):
        '''perform softmax operation by masking element on the paragraph dimension'''
        # mask contian 1 or 0
        # the shape of mask shoube be equal to num_para_1 * num_para_2
        # mask --> (#batch, num_para, num_para_bf)
        # scores --> (#batch, #num_para_1 * num_para_2,  #num_words, #num_words)

        # expand mask
        mask = mask.reshape(scores.size(0),scores.size(1))
        mask = mask.expand(scores.size(0), scores.size(1), scores.size(2), scores.size(3))
        # mask --> (#batch, #num_para_1 * num_para_2,  #num_words, #num_words)
        score_mask = torch.multiply(scores, mask) # should be pairwise
        # score_mask --> (#batch, #num_para_1 * num_para_2,  #num_words, #num_words)
        score_mask = score_mask.reshape(config.batch_size, config.para_len,\
                                        config.para_len, config.wrd_len, config.wrd_len)
        # score_mask --> (#batch, #num_para_1, #num_para_2,  #num_words, #num_words)
        score_softmax = nn.functional.softmax(scores, dim=1)
        score_softmax = score_mask.reshape(scores.shape)
        # score_softmax --> (#batch, #num_para_1 * num_para_2,  #num_words, #num_words)
        return score_softmax




    def forward(self, queries, keys, values, mask):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`

        # queries -> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        # keys -> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        # socre -> queries * keys.transpose -> (#batch, #num_para_1 * num_para_2,  #num_words, #num_words)

        scores = torch.matmul(queries, keys.transpose(-2,-1)) / math.sqrt(d)
    
        self.attention_weights = self.masked_softmax(scores, mask) 
        # attention_weights --> (#batch, #num_para_1 * num_para_2,  #num_words, #num_words)

        # values --> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        return torch.multiply(self.dropout(self.attention_weights), values) # formula: Q * K.T * V
        # output --> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)



class Self_Attention(nn.Module):
    def __init__(self, config, mask,\
                bias=False, **kwargs):
        self.W_q = nn.Linear(config.query_size, config.emb_dim, bias=bias)
        self.W_k = nn.Linear(config.key_size, config.emb_dim, bias=bias)
        self.W_v = nn.Linear(config.value_size, config.emb_dim, bias=bias)
        self.mask = mask
        self.dotattention = DotProductAttention(config.dropout_rate)
        # self.W_o = nn.Linear(dimension, dimension, bias=bias)

    def att_transpose(self, config, diff_metrix):
        # diff_metrix --> (#batch, #dimension, #num_para_1, #num_para_2, #num_words)
        diff_new = diff_metrix.reshape(config.batch_size, config.emb_dim, \
                                       config.para_len*config.para_len, config.wrd_len)
        # diff_new --> (#batch, #dimension, #num_para_1 * num_para_2, #num_words)
        diff_new = torch.permute(diff_new, (0,3,4,1))
        # diff_new --> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        return diff_new
    
    def att_transpose_back(self, config, attention_w):
        # attention_w --> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        attention_new = attention_w.reshape(config.batch_size, config.para_len,
                                       config.para_len, config.wrd_len, config.emb_dim)
        # attention_new --> (#batch, #num_para_1, #num_para_2,  #num_words, #dimension)
        attention_new = torch.permute(attention_new, (0,4,1,2,3))
        # attention_new --> (#batch, #dimension, #num_para_1, #num_para_2,  #num_words)
        return attention_new

    def forward(self, config, diff_metrix, mask):
        diff_metrix = self.att_transpose(config, diff_metrix)
        # after transpose diff_metrix --> 
        # (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)

        # question: should I mask before W_q, W_k, W_v or after?
        # if I mask before, the number of parameters in  W is smaller
        queries = self.W_q(diff_metrix) # queries -> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        keys = self.W_k(diff_metrix) # keys -> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        values = self.W_v(diff_metrix) # values -> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        attention_w = self.dotattention(queries, keys, values, mask)
        # attention_w --> (#batch, #num_para_1 * num_para_2,  #num_words, #dimension)
        attention_w = self.att_transpose_back(config, attention_w)
        # attention_w --> (#batch, #num_para_1, #num_para_2,  #num_words, #dimension)
        return attention_w


# define model
class simple_siamese(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_dim = config.emb_dim
        self.wrd_len = config.wrd_len
        self.para_len = config.para_len
        self.num_filters = config.num_filters
        self.kernel_sizes = config.kernel_sizes
        self.kernel_sizes2 = config.kernel_sizes2
        self.kernel_sizes3 = config.kernel_sizes3

        
        self.dropout_rate = config.dropout_rate
        self.num_classes = config.num_classes
        self.test_mode = config.test_mode

        #shrink on the paragraph
        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size = self.kernel_sizes), # input (#batch, #dimension, #num_para, #num_words)
            nn.Conv2d(128, 64,  kernel_size = self.kernel_sizes1), # input (#batch, #dimension, #num_para, #num_words)
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,3), padding=0),  # input (#batch, 128, 30, 40) #output (#batch, 128, 30, 13)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32,  kernel_size = self.kernel_sizes2), # input (#batch, #dimension, num_para, num_words) # kernal size = 10  # output: (#batch, 128, 30, 40)
            nn.Conv2d(32, 16,  kernel_size = self.kernel_sizes3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), padding=0),
        )

        linear_size = 64
        self.fc1 = nn.Linear(1056, linear_size)
        self.fc2 = nn.Linear(linear_size, int(self.num_classes))
        self.norl = nn.BatchNorm1d(linear_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
    def para_mask(self, valid_len, valid_len_bf):
        mask_1 = torch.arange(self.para_len)
        mask_2 = torch.arange(self.para_len)

        # arange start from 0
        # valid_len should be substruct 1
        para_mask = torch.where(mask_1 > valid_len-1, 0, 1)
        para_mask_bf = torch.where(mask_1 > valid_len_bf-1, 0, 1)

        mask = para_mask.multiply(para_mask_bf.reshape(-1,1)) # (num_para, num_para_bf)
        
        return mask


    def forward(self, input1, input2, valid_len, valid_len_bf):
        #calculate mask 
        mask = self.para_mask(valid_len, valid_len_bf)

        #permute input
        x1 = torch.permute(input1, (0,3,1,2)) #batch, #dimension, #num_para, #num_words 
        x2 = torch.permute(input2, (0,3,1,2))

        #calculate similarity between all pairs of paragraphs
        

        

        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        if self.test_mode:
            print('---conv1 output---')
            print(x1.size())
            print(x2.size())

        #change shape
        x1 = x1.unsqueeze(x1,3) # input (#batch, #dimension, #num_para, #num_words) --> #output (#batch, #dimension, #num_para, 1, #num_words)
        x2 = x2.unsqueeze(x1,2) # input (#batch, #dimension, #num_para, #num_words) --> #output (#batch, #dimension, 1, #num_para, #num_words)

        #get difference metrix
        x = torch.abs(torch.sub(x1,x2)) # -->(#batch, #dimension, #num_para_2, #num_para_1, #num_words)

        # attention
        attention_calculate = Self_Attention(self.config, mask)
        attention_weight = attention_calculate(self.config, x)
        # attention_weight -> (#batch, #num_para_1, #num_para_2,  #num_words, #dimension)
        
        # print(x.size())
        x = self.conv2(x)
        if self.test_mode:
            print('---conv2 output---')
            print(x.size())
        
        x = torch.reshape(x,(x.size()[0],-1))
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        # x = self.norl(x)
        logit = self.fc2(x)
        # print('model output',logit.size())

        x1 = torch.reshape(x1,(x1.size()[0],-1))
        x2 = torch.reshape(x2,(x2.size()[0],-1))

        return logit, x1, x2   
