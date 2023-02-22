
import torch
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, multilabel_confusion_matrix, \
                            f1_score, accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn as nn
import random
import utils
import copy
import pickle

def model_eval(model, config, validation_dataloader):
    #tokenized_texts = []
    true_labels = []
    pred_labels = []

    threshold = 0.5

    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        # print('val 1 free gpu',get_free_gpu())
        b_input_key = batch[0]
        b_labels = batch[1].to(config.device)


        #convert key to text embedding
        tk_batch = []
        tk_batch_bf = []
        #print('val batch',batch)
        for t in b_input_key.detach().to('cpu').numpy():
            tk, tk_bf = utils.get_text_embedding(t[0], t[1], t[2], config)
            if tk.size()[0] == config.para_len:              
                tk_batch.append(tk)
                tk_batch_bf.append(tk_bf)
            else:
                print('token size error')
                break
            

        tk_batch = torch.stack(tk_batch)
        tk_batch = tk_batch.to(config.device)

        tk_batch_bf = torch.stack(tk_batch_bf)
        tk_batch_bf = tk_batch_bf.to(config.device)
        # print('val 2 free gpu',get_free_gpu())

        with torch.no_grad():
            print('input shape',tk_batch.shape, tk_batch_bf.shape)
            logits, x1, x2 = model(tk_batch, tk_batch_bf)
            cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos_sim(x1,x2)
            sim = sim.reshape(-1,1)
            #loss_func = BCELoss()
            #val_loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation

            tk_batch.detach().to('cpu')
            del tk_batch
            tk_batch_bf.detach().to('cpu')
            del tk_batch_bf           
            # print('val 3 free gpu',get_free_gpu())
            
            if config.class_weight != None:
                pos_weight = torch.tensor(config.class_weight).to(config.device)
                # weights = torch.tensor([pos_weight]).to(device)
                sample_weight = np.array([config.class_weight if i[1] == 1 else 1 for i in b_labels])
                ct_loss = nn.CrossEntropyLoss() #weight = weights
                loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                ct_loss = nn.CrossEntropyLoss()
                loss_func = nn.BCEWithLogitsLoss()

            
            if config.set_ct_loss == True:
                val_loss =  loss_func(logits,b_labels.type_as(logits)) \
                    -  ct_loss(sim, torch.argmax(b_labels,axis=1).type_as(sim).reshape(-1,1))  #convert labels to float for calculation
            else: 
                val_loss =  loss_func(logits,b_labels.type_as(logits))

            total_eval_loss += val_loss.item()
            

            pred_label = torch.softmax(logits, dim=1)
            b_labels = b_labels.to('cpu').numpy()
            pred_label = pred_label.to('cpu').numpy()

            #tokenized_texts.append(b_input_ids)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = np.vstack(pred_labels)
    true_labels = np.vstack(true_labels)

    pred_bools = np.argmax(pred_labels, axis=1)

    avg_val_loss = total_eval_loss / len(validation_dataloader)


    pred_bools = np.argmax(pred_labels, axis=1)
    true_bools = np.argmax(true_labels, axis=1)

    val_f1 = f1_score(true_bools, pred_bools, average=None) * 100
    val_f1 = val_f1[1]  # return f1 for  class 1
    val_acc = (pred_bools == true_bools).astype(int).sum() / len(pred_bools)
    val_auc = roc_auc_score(true_bools, pred_labels[:,1])

    return  pred_labels, true_labels, avg_val_loss, val_f1, val_acc, val_auc


def train_model(model, config,  train_dataloader, validation_dataloader, 
                model_name, optimizer=None, scheduler=None, epochs = 20, \
                             patience = 5):

    utils.set_seed()

    threshold = 0.5
    #model_path = 'best_model.model'  # save the best model

    para_len = config.para_len
    wrd_len = config.wrd_len
    para_map = config.para_map
    class_weight = config.class_weight
    num_labels = config.num_labels
    verbose_mode = config.verbose_mode

    training_stats = []

    best_score = -0.5
    best_epoch = 0
    cnt = 0

    total_t0 = time.time()

    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
        train_true_labels = []
        train_pred_labels = []
        for step, batch in enumerate(train_dataloader):
            
            # `batch` contains three pytorch tensors:
            #   [0]: (cik, fyear, fyear_bf)
            #   [1]: labels
            
            # print('1 free gpu',get_free_gpu())
            b_input_key = batch[0] # batch_size * (cik, fyear, fyear_bf)
            b_labels = batch[1].to(config.device)
            
            
            #convert key to text embedding
            tk_batch = []
            tk_batch_bf = []
            #print('b_input_key',b_input_key)
            time_start_tk = time.time()
            for t in b_input_key.detach().to('cpu').numpy():
                tk, tk_bf = utils.get_text_embedding(t[0], t[1], t[2], config)
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
            tk_batch = tk_batch.to(config.device)
            
            tk_batch_bf = torch.stack(tk_batch_bf)
            tk_batch_bf = tk_batch_bf.to(config.device)
            #  print('2 free gpu',get_free_gpu())
            

            time_start_batch_train = time.time()
            logits, x1, x2 = model(tk_batch,tk_batch_bf)
            cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos_sim(x1,x2)
            sim = sim.reshape(-1,1)
            #print("logits shape: ", b_input_ids.size(), b_labels.size(), logits.shape())
            #loss_func = BCELoss()
            #loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation

            # add class weight
            if config.class_weight != None:
                pos_weight = torch.tensor(config.class_weight).to(config.device)
                sample_weight = np.array([config.class_weight if i[1] == 1 else 1 for i in b_labels])
                ct_loss = nn.CrossEntropyLoss()#weight = weights
                loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                ct_loss = nn.CrossEntropyLoss()
                loss_func = nn.BCEWithLogitsLoss()
            
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
            # global my_ct_loss, my_sim, my_label
            # my_sim = sim
            # my_label = b_labels
            # my_ct_loss = ct_loss(sim, torch.argmax(b_labels,axis=1).type_as(logits).reshape(-1,1)) 

            if config.verbose_mode:
                # print("logits: ", logits)
                # print("b_labels.type_as(logits): ", b_labels.type_as(logits))
                
                train_pred_bools = torch.argmax(logits, axis=1)
                train_pred_bools = train_pred_bools.to('cpu').numpy()
                train_true_bools = torch.argmax(b_labels.type_as(logits), axis=1)
                train_true_bools = train_true_bools.to('cpu').numpy()
                # print(train_pred_bools.shape, train_true_bools.shape)

                train_true_labels += train_true_bools.tolist()
                train_pred_labels += train_pred_bools.tolist()
                # print("train_pred_bools", train_pred_bools)
                # print("train_true_bools", train_true_bools)
                

            if config.set_ct_loss == True:
                loss =  loss_func(logits,b_labels.type_as(logits)) \
                    -  ct_loss(sim, torch.argmax(b_labels,axis=1).type_as(sim).reshape(-1,1))  #convert labels to float for calculation
            else: 
                loss =  loss_func(logits, b_labels.type_as(logits))

            total_train_loss += loss.item()
            # print(f"train step loss: {step} -- {loss}")
            # print(f"train step total_train_loss: {step} -- {total_train_loss}")

            model.zero_grad()
            
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()

            # Update the learning rate.
            if scheduler != None:
                scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = time.time() - t0
        print("Total training_time took {0:.2f} minutes ".format(training_time/60))

        # calculate the total accrurcy in this epoch
        global lista
        global listb
        lista = train_true_labels
        listb = train_pred_labels
        train_true_labels =  np.array(train_true_labels)
        train_pred_labels = np.array(train_pred_labels)
        print('training acc', (train_true_labels == train_pred_labels).sum(),len(train_true_labels) )
        train_acc = (train_true_labels == train_pred_labels).sum()/len(train_true_labels)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        testing = True

        if testing:
            print("")
            print("Running Validation...")

            t1 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            pred_labels, true_labels, avg_val_loss, val_f1, val_acc, val_auc = model_eval(model,config,\
                                                                                    validation_dataloader)

            # global val_label_save
            # global val_true_label_save
            # val_label_save.append(pred_labels)
            # val_true_label_save.append(true_labels)

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
                'Valid. AUC':val_auc,
                'Best F1': best_score,
                'Best epoch': best_epoch
                #'Training Time': training_time,
                #'Validation Time': validation_time
            })

            # early stopping
            if val_f1 > best_score:
                best_score = val_f1
                best_epoch = epoch_i + 1
                torch.save(copy.deepcopy(model.state_dict()), model_name)
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

def training_loop(config, mymodel):



    set_ct_loss = False
    result = []
    val_label_save = []
    val_true_label_save = []
    label_cols = ['fraud']

    # global my_ct_loss
    # global my_sim
    # global my_label

    #embedding
    print('Loading BERT tokenizer...')
    tokenizer = config.tokenizer
    bert_model = config.bert_model

    for col in label_cols:
        print("\n------------")
        print(col)
        print("------------")

        y = config.df[col].astype(int).values
        x_key = config.df[['cik', 'fyear', 'fyear_bf']].values

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
                batch_size = config.batch_size  # Trains with this batch size.
            )

            validation_dataloader = DataLoader(
                val_dataset,  # The validation samples.
                sampler=RandomSampler(
                    val_dataset),  # Pull out batches sequentially.
                batch_size= config.batch_size  # Evaluate with this batch size.
            )

            # if config.class_weight == None:
            #     pass
            # else:
            #     train_sample_weight = np.array(
            #         [config.class_weight if i[1] == 1 else 1 for i in Y_train])
            #     test_sample_weight = np.array(
            #         [config.class_weight if i[1] == 1 else 1 for i in Y_test])
            #  print('ready to build model')
            model_name = config.model_path + str(fold)
            # #model = cnn(emb_dim, seq_len, num_filters, kernel_sizes, num_labels)
            # model = mymodel(config)
            mymodel.to(config.device)
            # print('successfully build model')


            model, training_stats = train_model(mymodel, config, train_dataloader, validation_dataloader, \
                                                model_name = model_name)

            print("load the best model ... ")

            model.load_state_dict(torch.load(model_name))

            # show performance of best model
            model.eval()
            pred_labels, true_labels, avg_val_loss, val_f1, val_acc, val_auc = model_eval(model, config, validation_dataloader)
            


            pred_bools = np.argmax(pred_labels, axis = 1)
            print("np.argmax for pred_labels", pred_labels)
            true_bools = np.argmax(true_labels, axis = 1)
            print("np.argmax for true_labels", true_labels)

            p, r, f, _ = precision_recall_fscore_support(true_bools,pred_bools, pos_label = 1)
            #val_f1 = f1_score(true_bools,pred_bools, average = None)*100
            #val_f1 = val_f1[1] # return f1 for  class 1
            val_acc = (pred_bools == true_bools).astype(int).sum()/len(pred_bools)
            val_auc = roc_auc_score(true_bools, pred_labels[:,1])

            print('Precision: {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, Loss: {3:.4f}, AUC: {4:.4f}'.format(p[1], r[1], f[1], avg_val_loss, val_auc))
            print(classification_report(true_bools, pred_bools) )


            result.append([col, fold, p[1], r[1], f[1], val_acc, val_auc,training_stats[-1]["Best epoch"]])
            with open("/research/jujun/text_change/result/simple_siamese.pkl", "wb") as fp:   #Pickling
                pickle.dump(result, fp)
            
            torch.cuda.empty_cache()
            
    print('=== finish  === ')    
    return result, val_label_save, val_true_label_save