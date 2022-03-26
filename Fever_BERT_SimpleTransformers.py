from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from datasets import load_dataset
#from Document_Retrieval.GENRE_Document_Retrieval import RetrieveWiki
import numpy as np

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

fever = load_dataset("fever", "v1.0")

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

train_data = fever['train']
eval_data = fever['labelled_dev']

class BERTBinaryClassification:
    def __init__(self):
        self.model_args = ClassificationArgs(num_train_epochs=1,output_dir="E:/outputs_bin")
        self.model = ClassificationModel(
            "roberta",
            "roberta-base",
            num_labels=2,
            weight=[0.4, 1],
            args=self.model_args
        )
        
    def process_binary(self, dataset):
        dataframe = pd.DataFrame(dataset)
        dataframe = dataframe[['claim', 'label']]
        dataframe['label'] = dataframe['label'].apply(lambda x: convert_label_to_int(x))
        dataframe.drop(dataframe.index[dataframe['label'] == 2], inplace=True)
        dataframe.rename({'claim': 'text', 'label': 'labels'}, axis=1, inplace=True)
        return dataframe

    def train_and_eval(self, train_data, eval_data):
        train_data = self.process_binary(train_data)
        eval_data = self.process_binary(eval_data)
        self.model.train_model(train_data)
        result, model_outputs, wrong_predictions = self.model.eval_model(eval_data)
        return result

class BERTMultiClassification:
    def __init__(self):
        self.model_args = ClassificationArgs(num_train_epochs=1, output_dir="E:/outputs_mult")
        self.model = ClassificationModel(
            "roberta",
            'E:\outputs_mult\checkpoint-38929-epoch-1',
            num_labels=3,
            weight=[0.25, 0.65, 1],
            args=self.model_args
        ) 
    
    def process_multiclass(self, dataset):
        dataframe = pd.DataFrame(dataset)
        dataframe = dataframe[['claim', 'label']]
        dataframe['label'] = dataframe['label'].apply(lambda x: convert_label_to_int(x))
        dataframe.rename({'claim': 'text', 'label': 'labels'}, axis=1, inplace=True)
        return dataframe
    
    def train_and_eval(self, train_data, eval_data):
        train_data = self.process_multiclass(train_data)
        eval_data = self.process_multiclass(eval_data)
        self.model.train_model(train_data)
        result, model_outputs, wrong_predictions = self.model.eval_model(eval_data)
        return result
    

def build_format(dataset, predictions, gen_evidence):
    output = []
    for i in range(len(dataset)):
        evidence = dataset['evidence'][i]
        for j in range(len(dataset['evidence'][i])):
            for k in range(len(dataset['evidence'][i][j])):
                if dataset['evidence'][i][j][k][2]:
                    dataset['evidence'][i][j][k][2] = dataset['evidence'][i][j][k][2].lower()
        label = dataset['label'][i]
        predicted_label = convert_int_to_label(predictions[i])
        predicted_evidence = gen_evidence[i]
        output.append({'label':label, 'predicted_label':predicted_label,'predicted_evidence':predicted_evidence,'evidence': evidence})
    return output

def generate_evidence(dataset):
    from Document_Retrieval.GENRE_Document_Retrieval import RetrieveWiki
    import pickle
    import random

    with open("../wiki_dictionary.pkl", "rb") as f:
        wiki_dict = pickle.load(f)
    
    
    sent_selector = SentenceSelector()
    ret_wiki = RetrieveWiki()
    output = []
    
    documents_top_evidence = ret_wiki.generate_top_evidence(list(dataset['claim']))
    
    for i, sentence in enumerate(documents_top_evidence):
        try:
            sentence_evidence = wiki_dict[sentence]
            sentence_len = len(sentence_evidence)
            chosen_sentences = sent_selector.predict_sentences(dataset['claim'][i], sentence_evidence)
            output.append([[sentence, x] for x in chosen_sentences])
        except:
            output.append([[None, None]]*5)
            print("failed:", sentence)

    return output

from fever.scorer import fever_score
import sentence_selection_utils
import torch
from torch import nn

class SentenceSelector:
    def __init__(self):
        self.bert_model = "bert-base-uncased"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
        self.maxlen = 128  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
        self.bs = 16  # batch size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sentence_selection_utils.set_seed(42)
        self.path_to_model = 'models/bert-base-uncased_lr_2e-05_val_loss_0.33999_ep_1.pt'  
        self.path_to_output_file = 'results/output.txt'
    
    def build_data(self, claim, evidence):
        input_format = [[claim, e] for e in evidence]
        output_frame = pd.DataFrame(input_format, columns=['claim','evidence'])
        return output_frame
        
    def predict_sentences(self, claim, evidence):
        dataset = self.build_data(claim, evidence)
        test_set = sentence_selection_utils.CustomDataset(dataset, self.maxlen, self.bert_model)
        test_loader = sentence_selection_utils.DataLoader(test_set, batch_size=self.bs, num_workers=5)
        
        model = sentence_selection_utils.SentencePairClassifier(self.bert_model)
        if torch.cuda.device_count() > 1:  # if multiple GPUs
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        
        print()
        print("Loading the weights of the model...")
        self.model.load_state_dict(torch.load(self.path_to_model))
        self.model.to(self.device)
        
        print("Predicting on test data...")
        probs = sentence_selection_utils.test_prediction(net=self.model, device=self.device, dataloader=test_loader, with_labels=True, result_file=self.path_to_output_file)
        
        indeces = np.argpartition(probs, -5)[-5:]
        return indeces

if __name__ == "__main__":
    df_train = pd.read_json(path_or_buf='./task_train.jsonl', lines=True).drop(['id'], axis=1)
    df_test = pd.read_json(path_or_buf='./task_test.jsonl', lines=True).drop(['id'], axis=1)
    df_val = pd.read_json(path_or_buf='./task_dev.jsonl', lines=True).drop(['id'], axis=1)
    
    print(df_train.shape)
    print(df_test.shape)
    print(df_val.shape)
    
    #BERT_bin = BERTBinaryClassification()
    BERT_multi = BERTMultiClassification()
    
    #print(BERT_bin.train_and_eval(train_data, eval_data))
    #print(BERT_multi.train_and_eval(train_data, eval_data)
    
    #test_data = BERT_multi.process_multiclass(df_val)
    outs = BERT_multi.model.predict(list(df_test['claim']))
    
    del BERT_multi
    
    #predict_sentences(df_test)
    
    gen_evidence = generate_evidence(df_test)
    output_format = build_format(df_test, outs[0], gen_evidence)
    
    strict_score, label_accuracy, precision, recall, f1 = fever_score(output_format)    
    
    
    
    
