import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric
from sklearn.utils import resample
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'claim'])
        sent2 = str(self.data.loc[index, 'evidence'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        output = self.bert_layer(input_ids, attn_masks, token_type_ids)
        
        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        
        logits = self.cls_layer(self.dropout(output[1]))

        return logits

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count

print("Creation of the models' folder...")


"""Link for mixed precision training, gradient scaling and gradient accumulation  : https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples

If you would like to learn more about Training Neural Nets on Larger Batches, I suggest reading this post of Thomas Wolf :
https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
"""

def train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()


            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model='models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()


def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()

def convert_label_to_int(label):
  if label == 'SUPPORTS':
    return 0
  elif label == 'REFUTES':
    return 1
  else:
    return 2

def convert_int_to_label(int_val):
    if int_val == 0:
        return "SUPPORTS"
    elif int_val == 1:
        return "REFUTES"
    else:
        return "NOT ENOUGH INFO"

def process_binary(self, dataset):
    dataframe = pd.DataFrame(dataset)
    dataframe = dataframe[['claim', 'label']]
    dataframe['label'] = dataframe['label'].apply(lambda x: convert_label_to_int(x))
    dataframe.drop(dataframe.index[dataframe['label'] == 2], inplace=True)
    dataframe.rename({'claim': 'text', 'label': 'labels'}, axis=1, inplace=True)
    return dataframe

if __name__ == "__main__":
    
    df_train = pd.read_csv("train.csv", on_bad_lines="warn", engine="python", encoding='utf-8', index_col=0)
    df_val = pd.read_csv("test.csv", on_bad_lines="warn", engine="python",encoding='utf-8', index_col=0)
    df_test = pd.read_csv("dev.csv", on_bad_lines="warn", engine="python",encoding='utf-8', index_col=0)
    
    df_majority = df_train[df_train.label == 0]
    df_minority = df_train[df_train.label == 1]
    df_maj_downsample = resample(df_majority, replace=False, n_samples=127065, random_state=423)
    df_downsampled = pd.concat([df_maj_downsample, df_minority])
    df_train = df_downsampled
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    
    df_majority = df_test[df_test.label == 0]
    df_minority = df_test[df_test.label == 1]
    df_maj_downsample = resample(df_majority, replace=False, n_samples=6200, random_state=423)
    df_downsampled = pd.concat([df_maj_downsample, df_minority])
    df_test = df_downsampled
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)
    
    print(df_train.head())
    
    bert_model = "bert-base-uncased"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
    freeze_bert = True  # if True, freeze the encoder weights and only update the classification layer weights
    maxlen = 128  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
    bs = 16  # batch size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 2e-5  # learning rate
    epochs = 3  # number of training epochs
    
    #  Set all seeds to make reproducible results
    set_seed(42)
    
    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train, maxlen, bert_model)
    print("Reading validation data...")
    val_set = CustomDataset(df_val, maxlen, bert_model)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=8)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)
    
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    
    net.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    #train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)

    path_to_model = 'models/bert-base-uncased_lr_2e-05_val_loss_0.56412_ep_3.pt'  

    path_to_output_file = 'results/output.txt'
    
    print("Reading test data...")
    test_set = CustomDataset(df_test, maxlen, bert_model)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)
    
    model = SentencePairClassifier(bert_model)
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    
    print("Predicting on test data...")
    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))

    path_to_output_file = 'results/output.txt'  # path to the file with prediction probabilities
    
    labels_test = df_test['label']  # true labels
    
    probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
    threshold = 0.5   # you can adjust this threshold for your own dataset
    preds_test=(probs_test>=threshold).astype('uint8') # predicted labels using the above fixed threshold
    
    metric = load_metric("glue", "mrpc")

    # Compute the accuracy and F1 scores
    print(metric._compute(predictions=preds_test, references=labels_test))
    
    
    
