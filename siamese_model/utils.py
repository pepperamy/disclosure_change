
import torch
import pandas as pd
import numpy as np



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
