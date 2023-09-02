"""
@author: Firoj Alam
@email: firojalam@gmail.com
Modified: 
"""

import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import logging
import json
import os
from torch import nn
import performance as performance
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import sklearn.metrics as metrics

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Reads the data and do some preprocessing
"""
def read_data(data_file):
    """
    Prepare the data
    """
    embedding = []
    labels = []
    with open(data_file, encoding='utf-8') as gold_f:
        i = 0
        for line in gold_f:
            if line.strip() != '':
                row_dict = json.loads(line)
                #print(row_dict)
                i+=1
                #if i >= 19570:
                #    print(row_dict)
                if 'response' in row_dict:
                    emb = row_dict["response"]["data"][0]["embedding"]
                else:
                    emb = row_dict["average_embedding"]
                #print(i)
                label = row_dict["label"]
                embedding.append(emb)
                labels.append(label)

    label_list=list(set(labels))
    label_list.sort()
    label_to_index={}
    index_to_label = {}
    for index, lab in enumerate(label_list):
        label_to_index[lab]=index
        index_to_label[index] = lab

    labels_rep=[]

    for l in labels:
        labels_rep.append(label_to_index[l])

    return np.array(embedding), np.array(labels_rep),label_to_index, index_to_label

class TaskDataset(pl.LightningDataModule):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length

class Data(pl.LightningDataModule):

    def __init__(self,batch_size,train_file="",dev_file="",tst_file=""):
        train_x, train_y, label_to_index, index_to_label = read_data(train_file)
        # dev_x, dev_y, , label_to_index, index_to_label = read_data(dev_file)
        test_x, test_y, label_to_index, index_to_label = read_data(tst_file)
        self.num_of_label = len(set(train_y))
        self.batch_size = batch_size
        self.train_data = TaskDataset(train_x, train_y)
        # self.dev_data = TaskDataset(dev_x, dev_y)
        self.test_data = TaskDataset(test_x, test_y)


    def train_dataloader(self):
        data_loader = DataLoader(dataset=self.train_data, shuffle=True, batch_size=self.batch_size)
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(dataset=self.dev_data, shuffle=True, batch_size=self.batch_size)
        return data_loader

    def test_dataloader(self):
        data_loader = DataLoader(dataset=self.test_data, shuffle=True, batch_size=self.batch_size)
        return data_loader




class LitNeuralNet(pl.LightningModule):
    def __init__(self,input_size, num_classes, hidden_size, learning_rate):
        super(LitNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        y_pred = self.forward(x)
        _,y_pred = y_pred.softmax(dim=-1)
        y_pred = torch.max(y_pred, 1)
        acc= metrics.accuracy_score(y, y_pred)
        self.log_dict({'val_loss': loss, "val_acc":acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = y_pred.softmax(dim=-1)
        predicted = torch.max(y_pred, 1)
        return predicted


def training(train_file, dev_file, tst_file, model_dir, results_dir):
    dirname = os.path.dirname(train_file)
    base = os.path.basename(train_file)
    file_name = os.path.splitext(base)[0]

    ## read data
    train_x, train_y,label_to_index, index_to_label = read_data(train_file)
    # dev_x, dev_y, _ = read_data(dev_file, delim, train_le, lang,label_index)
    test_x, test_y,label_to_index, index_to_label = read_data(tst_file)
    num_of_label = len(set(train_y))

    # Hyper-parameters
    input_size = 1536  # text-ada
    hidden_size = 500
    num_classes = num_of_label
    num_epochs = 1500
    batch_size = 100
    learning_rate = 0.001

    train_dataset = TaskDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

    test_dataset = TaskDataset(test_x, test_y)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    model = LitNeuralNet(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size, learning_rate=learning_rate)
    # %% Callbacks
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=20, verbose=True, mode="min")

    # training
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=num_epochs, callbacks=[early_stop_callback],
                         log_every_n_steps=8)
    trainer.fit(model=model, train_dataloaders=train_loader)
    # %% testing
    trainer.test(model=model, dataloaders=test_loader)



    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except Exception as e:
        pass
    best_model_path = model_dir + file_name + "_ffn.model"

    predictions = trainer.predict(model, test_loader)
    #print(predictions)

    all_prob=[]
    all_pred=[]
    for pred_prob, labels in predictions:
        all_prob.extend(pred_prob.tolist())
        all_pred.extend(labels)
        # print(pred_prob)

    AUC, acc, precision, recall, F1, report = performance.performance_measure(test_y,all_pred)
    result = str("{0:.4f}".format(acc)) + "\t" + str("{0:.4f}".format(precision)) + "\t" + str(
        "{0:.4f}".format(recall)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC)) + "\n"

    print ("Test set:\t" + result)
    print(report)

    return model

def save_model(model_dir, model_file_name, model):

    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    model_file = model_dir + "/" + base_name+"_"+ "_model.pth"


    ############save only state dict #########################
    # save only state dict
    torch.save(model.state_dict(), model_file)

    # print(model.state_dict())
    # loaded_model = Model(n_input_features=6)
    # loaded_model.load_state_dict(torch.load(FILE))  # it takes the loaded dictionary, not the path file itself
    # loaded_model.eval()

    # print(loaded_model.state_dict())


if __name__ == '__main__':
    print('v1.01.01')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", required=True, type=str,
                        help="Path to file containing taining data.")
    parser.add_argument("--dev", "-d", required=True, type=str,
                        help="Path to file containing dev data.")
    parser.add_argument("--test", "-s", required=True, type=str,
                        help="Path to file containing test data.")
    parser.add_argument("--model-dir", "-m", required=True, type=str,
                        help="Path to dirctory containing model.")
    parser.add_argument("--results-dir", "-r", required=True, type=str,
                        help="Path to dirctory containing model.")

    args = parser.parse_args()
    train_file = args.train
    dev_file = args.dev
    test_file = args.test

    model_dir = args.model_dir
    results_dir = args.results_dir

    print ("{} {} {} {} {}".format(train_file, dev_file, test_file, model_dir, results_dir))
    model=training(train_file, dev_file, test_file, model_dir, results_dir)

    dirname = os.path.dirname(train_file)
    base = os.path.basename(train_file)
    file_name = os.path.splitext(base)[0]
    save_model(model_dir, file_name, model)