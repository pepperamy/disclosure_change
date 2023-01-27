# %%
# initialization
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import sys, os
cur_path = os.path.join('/research/jujun/text_change')
os.chdir(cur_path)

import random, pickle
import numpy as np
from torch.nn import BCEWithLogitsLoss, BCELoss, MSELoss
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
# import tensorflow as tf
import torch
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from itertools import cycle
from tqdm import tqdm
import time
import copy
import datetime
# from numba import cuda 

# from pynvml import *
def get_free_gpu():
    print('\n')
    # nvmlInit()
    # h = nvmlDeviceGetHandleByIndex(0)
    # info = nvmlDeviceGetMemoryInfo(h)
    # print(f'total    : {info.total // 1024 ** 2}')
    # print(f'free     : {info.free// 1024 ** 2}')
    # print(f'used     : {info.used// 1024 ** 2}')

# %%
def get_pretrained_wordvector(sentences, tokenizer, bert_model, max_len=100):

    input_ids = []
    attention_masks = []
    max_len = max_len

    # For every sentence...
    for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        #padding='max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])


    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids.to(device), attention_masks.to(device))   
        hidden_states = outputs[2]

    
    # get the last four layers
    token_embeddings = torch.stack(hidden_states[-4:], dim=0) 
    #print(token_embeddings.size())

    # permute axis
    token_embeddings = token_embeddings.permute(1,2,0,3)
    #print(token_embeddings.size())

    # take the mean of the last 4 layers
    token_embeddings = token_embeddings.mean(axis=2)

    #print(token_embeddings.size())
    input_ids.detach().to('cpu')
    attention_masks.detach().to('cpu')
    token_embeddings.detach().to('cpu')
    del input_ids
    return token_embeddings, attention_masks

# %%
def get_text_embedding(cik, fyear, fyear_bf, tokenizer, bert_model, para_map, para_len, wrd_len=100):
    # print(cik, fyear, fyear_bf)
    df = pd.concat({k: pd.Series(v) for k, v in para_map[cik].items()})
    df = df.reset_index()
    df.columns = ['fyear','pid','text']

    input = df[df.fyear == fyear].text.values
    input_bf = df[df.fyear == fyear_bf].text.values

    #get embedding for input
    token_embeddings, masks = get_pretrained_wordvector(input, tokenizer, bert_model, max_len = wrd_len)
    token_embeddings = token_embeddings.to(device) * masks.unsqueeze(-1).to(device) # (atc_num_para, #wrd_len, #dim)
    # padding paragraphs
    # print('1 token_embeddings',token_embeddings.size())
    pad_num = para_len - token_embeddings.size()[0]
    if pad_num>0:
        token_embeddings = F.pad(input=token_embeddings, pad=(0,0,0,0,0,pad_num))
        # print('2 token_embeddings',token_embeddings.size())
    elif pad_num<0:
        token_embeddings = token_embeddings[0:para_len]
        # print('2 token_embeddings',token_embeddings.size())
    else:
        token_embeddings = token_embeddings

    #get embedding for input_bf
    token_embeddings_bf, masks_bf = get_pretrained_wordvector(input_bf, tokenizer, bert_model, max_len = wrd_len)
    token_embeddings_bf = token_embeddings_bf.to(device) * masks_bf.unsqueeze(-1).to(device) # (atc_num_para, #wrd_len, #dim)
    # padding paragraphs
    # print('1 token_embeddings_bf',token_embeddings_bf.size())
    pad_num_bf = para_len - token_embeddings_bf.size()[0]
    #print('pad_num_bf', pad_num_bf)
    if pad_num_bf>0:
        # print('>0')
        token_embeddings_bf = F.pad(input=token_embeddings_bf, pad=(0,0,0,0,0,pad_num_bf))
        # print('2 token_embeddings_bf',token_embeddings_bf.size())
    elif pad_num_bf<0:
        # print('<0')
        token_embeddings_bf = token_embeddings_bf[0:para_len]
        # print('2 token_embeddings_bf',token_embeddings_bf.size())
    else:
        token_embeddings_bf = token_embeddings_bf

    return token_embeddings, token_embeddings_bf


# %%
# define model
class simple_siamese(nn.Module):
    def __init__(self, emb_dim, wrd_len, num_filters, kernel_sizes, kernel_sizes2, num_classes=2.0, dropout_rate = 0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.wrd_len = wrd_len
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.kernel_sizes2 = kernel_sizes2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size = kernel_sizes), # input (#batch, 768, num_para->60, num_words->100) # kernal size = 10,50  # output: (#batch, 256, 50, 50)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, padding=0),  # input (#batch, 256, 50, 50) #output (#batch, 256, 10, 10)
            nn.Conv2d(128, 64,  kernel_size = kernel_sizes2), # input (#batch, 256, num_para->10, num_words->10) # kernal size = 3,3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.ReLU(), 
        )

        self.fc = nn.Linear(1152, int(self.num_classes))
        self.dropout = nn.Dropout(p=dropout_rate)
        

    def forward(self, input1, input2):
        #permute input to make it fit cnn
        x1 = torch.permute(input1, (0,3,1,2))
        x2 = torch.permute(input2, (0,3,1,2))
        # print(x1.size())
        # print(x2.size())

        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x = torch.sub(x1,x2)

        
        # print(x.size())
        x = torch.reshape(x,(x.size()[0],-1))
        # print(x.size())
        x = self.dropout(x)
        logit = self.fc(x)
        # print('model output',logit.size())

        return logit   


# %% [markdown]
# 

# %%
emb_dim = 768
wrd_len = 50 #100
para_len = 30 #60
num_filters = 128
kernel_sizes =  (10,10)
kernel_sizes2 =  (1,1) #(2,2)
dropout_rate = 0.5
num_classes=2.0
batch_size = 32
# para_map = para_map
class_weight = 1
model = simple_siamese( emb_dim, wrd_len, num_filters, kernel_sizes, \
    kernel_sizes2, num_classes=num_classes,dropout_rate=dropout_rate)
#summary(model [(32, 30, 50, 768), (32, 30, 50, 768)])

# %%
summary(model, [(32, 30, 50, 768), (32, 30, 50, 768)])

# %%
def model_eval(model, validation_dataloader, num_labels, class_weight=None):
    #tokenized_texts = []
    true_labels = []
    pred_labels = []

    threshold = 0.5

    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        # print('val 1 free gpu',get_free_gpu())
        b_input_key = batch[0]
        b_labels = batch[1].to(device)


        #convert key to text embedding
        tk_batch = []
        tk_batch_bf = []
        #print('val batch',batch)
        for t in b_input_key.detach().to('cpu').numpy():
            tk, tk_bf = get_text_embedding(t[0], t[1], t[2], tokenizer, bert_model, para_map, para_len, wrd_len=wrd_len)
            if tk.size()[0] == para_len:              
                tk_batch.append(tk)
                tk_batch_bf.append(tk_bf)
            else:
                print('token size error')
                break
            

        tk_batch = torch.stack(tk_batch)
        tk_batch = tk_batch.to(device)

        tk_batch_bf = torch.stack(tk_batch_bf)
        tk_batch_bf = tk_batch_bf.to(device)
        # print('val 2 free gpu',get_free_gpu())

        with torch.no_grad():

            logits = model(tk_batch, tk_batch_bf)
            #loss_func = BCELoss()
            #val_loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation

            tk_batch.detach().to('cpu')
            del tk_batch
            tk_batch_bf.detach().to('cpu')
            del tk_batch_bf           
            # print('val 3 free gpu',get_free_gpu())
            
            if class_weight != None:
                pos_weight = torch.tensor(class_weight).to(device)
                loss_func = BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_func = BCEWithLogitsLoss()

            val_loss = loss_func(
                logits,
                b_labels.type_as(logits))  #convert labels to float for calculation

            total_eval_loss += val_loss.item()

            pred_label = torch.sigmoid(logits)
            b_labels = b_labels.to('cpu').numpy()
            pred_label = pred_label.to('cpu').numpy()

            #tokenized_texts.append(b_input_ids)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = np.vstack(pred_labels)
    true_labels = np.vstack(true_labels)

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    return  pred_labels, true_labels, avg_val_loss

# %%
def train_model(model, num_labels, para_len, wrd_len, train_dataloader, validation_dataloader, model_path,\
                             optimizer=None, scheduler=None, epochs = 10, \
                             class_weight = None, patience = 5):

    seed_val = 42

    threshold = 0.5
    #model_path = 'best_model.model'  # save the best model

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    best_score = -0.5
    best_epoch = 0
    cnt = 0

    total_t0 = time.time()

    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            
            # `batch` contains three pytorch tensors:
            #   [0]: (cik, fyear, fyear_bf)
            #   [1]: labels
            
            # print('1 free gpu',get_free_gpu())
            b_input_key = batch[0] # batch_size * (cik, fyear, fyear_bf)
            b_labels = batch[1].to(device)
            
            
            #convert key to text embedding
            tk_batch = []
            tk_batch_bf = []
            #print('b_input_key',b_input_key)
            time_start_tk = time.time()
            for t in b_input_key.detach().to('cpu').numpy():
                tk, tk_bf = get_text_embedding(t[0], t[1], t[2], tokenizer, bert_model, para_map, para_len, wrd_len=wrd_len)
                if tk.size()[0] == para_len:              
                    tk_batch.append(tk)
                    tk_batch_bf.append(tk_bf)
                    # print(len(tk_batch), len(tk_batch_bf))
                else:
                    print('token size error')
                    break
            # print(len(tk_batch), len(tk_batch_bf))
            # print("----- token %s seconds -----" % (time.time() - time_start_tk))
                
            tk_batch = torch.stack(tk_batch)
            tk_batch = tk_batch.to(device)
            
            tk_batch_bf = torch.stack(tk_batch_bf)
            tk_batch_bf = tk_batch_bf.to(device)
            #  print('2 free gpu',get_free_gpu())
            model.zero_grad()

            time_start_batch_train = time.time()
            logits = model(tk_batch,tk_batch_bf)
            #print("logits shape: ", b_input_ids.size(), b_labels.size(), logits.shape())
            #loss_func = BCELoss()
            #loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation

            # add class weight
            if class_weight != None:
                pos_weight = torch.tensor(class_weight).to(device)
                loss_func = BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_func = BCEWithLogitsLoss()
            
            tk_batch.detach().to('cpu')
            del tk_batch
            tk_batch_bf.detach().to('cpu')
            del tk_batch_bf
            
            # print('3 free gpu',get_free_gpu())
            # print(logits.size(), b_labels.size())
#             loss = loss_func(
#                 logits.view(-1, num_labels),
#                 b_labels.type_as(logits).view(
#                     -1, num_labels))  
            # convert labels to float for calculation
            loss = loss_func(logits, b_labels.type_as(logits))
             
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()

            # Update the learning rate.
            if scheduler != None:
                scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = time.time() - t0
        print("Total training_time took {0:.2f} minutes ".format(training_time/60))

            #print("")
            #print("  Average training loss: {0:.2f}".format(avg_train_loss))
            #print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        testing = False

        if testing:
            print("")
            print("Running Validation...")

            t1 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            pred_labels, true_labels, avg_val_loss = model_eval(
                model,  validation_dataloader, num_labels, class_weight=class_weight)

            pred_bools = np.argmax(pred_labels, axis=1)
            true_bools = np.argmax(true_labels, axis=1)

            val_f1 = f1_score(true_bools, pred_bools, average=None) * 100
            val_f1 = val_f1[1]  # return f1 for  class 1
            val_acc = (
                pred_bools == true_bools).astype(int).sum() / len(pred_bools)

            #print('Validation Accuracy: {0:.4f}, F1: {1:.4f}, Loss: {2:.4f}'.format(val_f1, val_acc, avg_val_loss))
            #print(classification_report(np.array(true_labels), pred_bools, target_names=label_cols) )
            print("Epoch {0}\t Train Loss: {1:.4f}\t Val Loss {2:.4f}\t Val Acc: {3:.4f}\t Val F1: {4:.4f}".\
                format(epoch_i +1, avg_train_loss, avg_val_loss, val_acc, val_f1))

            # Measure how long the validation run took.
            validation_time = time.time() - t1
            print("Total val_time took {0:.2f} minutes ".format(validation_time/60))

            #print("  Validation Loss: {0:.2f}".format(val_f1_accuracy))
            #print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': val_f1,
                'Best F1': best_score,
                'Best epoch': best_epoch
                #'Training Time': training_time,
                #'Validation Time': validation_time
            })

            # early stopping
            if val_f1 > best_score:
                best_score = val_f1
                best_epoch = epoch_i + 1
                torch.save(copy.deepcopy(model.state_dict()), model_path)
                print("model saved")
                cnt = 0
            else:
                cnt += 1
                if cnt == patience:
                    print("\n")
                    print("early stopping at epoch {0}".format(epoch_i + 1))
                    break

            print("")
            #print("Training complete!")

            print("Total training took {0:.2f} minutes".format((time.time()-total_t0)/60))
        else:
            training_stats = 0
            print(avg_train_loss)
        
    return model, training_stats

# %%
if __name__ == "__main__":

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

    print(get_free_gpu())

    # load data
    para_map = pickle.load(open("/research/rliu/fraud/data/mda/paragraphs_1994_2016.pkl","rb"))
    pos_neg_pair = pd.read_csv('./data/pos_neg_pair.csv')
    pos_neg_pair = pos_neg_pair.dropna()

    print('successfully load data ...')

    pos_index = pos_neg_pair[pos_neg_pair.fraud == 1].index
    neg_index = pos_neg_pair[pos_neg_pair.fraud == 0].sample(len(pos_index)).index
    df = pos_neg_pair.loc[neg_index.append(pos_index),:]
    print(df.shape)


    emb_dim = 768
    wrd_len = 50 #100
    para_len = 30 #60
    num_filters = 128
    kernel_sizes =  (10,10)
    kernel_sizes2 =  (1,1) #(2,2)
    dropout_rate = 0.5
    num_classes=2.0
    batch_size = 16
    para_map = para_map
    class_weight = 1

    result = []
    label_cols = ['fraud']

    #embedding
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    bert_model = AutoModel.from_pretrained(
        # 'ProsusAI/finbert',
        'bert-base-uncased',
        # 'yiyanghkust/finbert-pretrain',
        num_labels = 2, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
        )
    bert_model.cuda()

    for col in label_cols:
        print("\n------------")
        print(col)
        print("------------")

        y = df[col].astype(int).values
        x_key = df[['cik', 'fyear', 'fyear_bf']].values

        fold = 0

        skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)

        for train_index, test_index in skf.split(x_key, y):

            print("\nfold {} \n".format(fold))

            fold += 1
            X_train, X_test = x_key[train_index], x_key[test_index]
            X_train = torch.tensor(X_train)
            X_test = torch.tensor(X_test)

            Y_train, Y_test = y[train_index], y[test_index]
            print('train fraud', sum(Y_train),'test fraud', sum(Y_test))

            Y_train = pd.get_dummies(Y_train).values
            Y_train = torch.tensor(Y_train)

            Y_test = pd.get_dummies(Y_test).values
            Y_test = torch.tensor(Y_test)

            train_dataset = TensorDataset(X_train, Y_train)
            val_dataset = TensorDataset(X_test, Y_test)

            train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler=RandomSampler(train_dataset),  # Select batches randomly
                batch_size=batch_size  # Trains with this batch size.
            )

            validation_dataloader = DataLoader(
                val_dataset,  # The validation samples.
                sampler=RandomSampler(
                    val_dataset),  # Pull out batches sequentially.
                batch_size=batch_size  # Evaluate with this batch size.
            )

            if class_weight == None:
                pass
            else:
                train_sample_weight = np.array(
                    [class_weight if i[1] == 1 else 1 for i in Y_train])
                test_sample_weight = np.array(
                    [class_weight if i[1] == 1 else 1 for i in Y_test])

            model_name = "./model/simple_siamese_" + str(fold)
            #model = cnn(emb_dim, seq_len, num_filters, kernel_sizes, num_labels)
            model = simple_siamese(emb_dim, wrd_len, num_filters, kernel_sizes,\
                                kernel_sizes2, num_classes=num_classes, dropout_rate = dropout_rate)
            model.to(device)


            model, training_stats = train_model(model, num_classes, para_len, wrd_len, train_dataloader, validation_dataloader, \
                                                            model_path = model_name, class_weight = class_weight,\
                                                            optimizer=None, scheduler=None, epochs = 20)

            print("load the best model ... ")

            model.load_state_dict(torch.load(model_name))

            # show performance of best model
            model.eval()
            pred_labels, true_labels,avg_val_loss = model_eval(model, \
                                                    validation_dataloader, num_classes, class_weight = class_weight)

            pred_bools = np.argmax(pred_labels, axis = 1)
            true_bools = np.argmax(true_labels, axis = 1)

            p, r, f, _ = precision_recall_fscore_support(true_bools,pred_bools, pos_label = 1)
            #val_f1 = f1_score(true_bools,pred_bools, average = None)*100
            #val_f1 = val_f1[1] # return f1 for  class 1
            val_acc = (pred_bools == true_bools).astype(int).sum()/len(pred_bools)

            print('Precision: {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, Loss: {3:.4f}'.format(p[1], r[1], f[1], avg_val_loss))
            print(classification_report(true_bools, pred_bools) )


            result.append([col, fold, p[1], r[1], f[1], training_stats[-1]["Best epoch"]])
            with open("./result/simple_siamese.pkl", "wb") as fp:   #Pickling
                pickle.dump(result, fp)
            
            torch.cuda.empty_cache()
            get_free_gpu()
    print('=== finish  === ')

# %%



