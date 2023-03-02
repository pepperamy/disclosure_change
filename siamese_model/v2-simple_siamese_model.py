

import torch.nn as nn
import torch
import math

class self_attention(nn.Module):
    def __init__(self, config, \
                  dropout, bias=False, **kwargs):
        self.q = nn.Linear(config.query_size, config.emb_dim, bias=bias)
        self.k = nn.Linear(config.key_size, config.emb_dim, bias=bias)
        self.v = nn.Linear(config.value_size, config.emb_dim, bias=bias)
        # self.W_o = nn.Linear(dimension, dimension, bias=bias)

    def att_transpose(self, config, diff_metrix):
        # diff_metrix --> (#batch, #dimension, #num_para_1, #num_para_2, #num_words)
        diff_new = diff_metrix.reshape(config.batch_size, config.emb_dim, \
                                       config.para_len*config.para_len, config.wrd_len)
        # diff_new --> (#batch, #dimension, #num_para_1 * num_para_2, #num_words)

        return diff_new

    def forward(self, config, diff_metrix, mask):

        diff_metrix = self.att_transpose(config, diff_metrix)

        


        d = self.q.shape[-1]
        scores = torch.bmm(self.q, self.k.transpose(1,2)) / math.sqrt(d)








# define model
class simple_siamese(nn.Module):
    def __init__(self, config):
        super().__init__()
        
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
