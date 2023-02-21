

import torch.nn as nn
import torch


# define model
class simple_siamese(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.emb_dim = config.emb_dim
        self.wrd_len = config.wrd_len
        self.num_filters = config.num_filters
        self.kernel_sizes = config.kernel_sizes
        self.kernel_sizes2 = config.kernel_sizes2
        self.kernel_sizes3 = config.kernel_sizes3
        self.dropout_rate = config.dropout_rate
        self.num_classes = config.num_classes
        self.test_mode = config.test_mode

        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size = self.kernel_sizes), # input (#batch, 768, num_para->30, num_words->50) # kernal size = 10  # output: (#batch, 128, 30, 40)
            nn.Conv2d(128, 64,  kernel_size = self.kernel_sizes2), # input (#batch, 768, num_para->30, num_words->50) # kernal size = 10  # output: (#batch, 128, 30, 40)
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,3), padding=0),  # input (#batch, 128, 30, 40) #output (#batch, 128, 30, 13)
            # nn.MaxPool1d(3, padding=0),  # input (#batch, 128, 30, 40) #output (#batch, 128, 30, 13)
            # nn.Conv2d(128, 64,  kernel_size = kernel_sizes2), # input (#batch, 256, num_para->10, num_words->10) # kernal size = 5
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d((1,1), padding=0),
            # nn.ReLU(), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32,  kernel_size = self.kernel_sizes2), # input (#batch, 768, num_para->30, num_words->50) # kernal size = 10  # output: (#batch, 128, 30, 40)
            nn.Conv2d(32, 16,  kernel_size = self.kernel_sizes3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), padding=0),
        )

        
        linear_size = 64
        self.fc1 = nn.Linear(1056, linear_size)
        self.fc2 = nn.Linear(linear_size, int(self.num_classes))
        self.norl = nn.BatchNorm1d(linear_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        

    def forward(self, input1, input2):
        #permute input to make it fit cnn
        x1 = torch.permute(input1, (0,3,1,2))
        x2 = torch.permute(input2, (0,3,1,2))
        # print(x1.size())
        # print(x2.size())

        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        if self.test_mode:
            print('---conv1 output---')
            print(x1.size())
            print(x2.size())
        x = torch.abs(torch.sub(x1,x2))

        
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
