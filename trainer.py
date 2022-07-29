# Built-in libraries
import copy
import datetime
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
# Local files
from multi_task_offensive_language_detection.utils import save
from multi_task_offensive_language_detection.config import LABEL_DICT

import wandb
class Trainer():
    '''
    The trainer for training models.
    It can be used for both single and multi task training.
    Every class function ends with _m is for multi-task training.
    '''
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        loss_weights: List[float],
        clip: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        patience: int,
        task_name: str,
        model_name: str,
        seed: int
    ):
        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.clip = clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.task_name = task_name
        self.model_name = model_name
        self.seed = seed
        self.datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.train_f1 = []
        self.test_f1 = []
        self.best_train_f1 = 0.0
        self.best_test_f1 = 0.0

        # Evaluation results for multi-task
        self.best_train_f1_m = np.array([0, 0, 0], dtype=np.float64)
        self.best_test_f1_m = np.array([0, 0, 0], dtype=np.float64)

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch(epoch=epoch)
            self.test()
            print(f'Best test f1: {self.best_test_f1:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1, self.best_test_f1),
            f'./save/results/single_{self.task_name}_{self.datetimestr}_{self.best_test_f1:.4f}.pt'
        )

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Testing'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                loss += _loss.item()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))
            wandb.log({"loss train epoch": loss})

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.save_model()

    def train_m(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            dataloader = self.dataloaders['train']
            #if np.mod(epoch, 3): ## Refresh hard samples.
            #    dataloader.dataset.hard_collection(percent_hard = 10)

            self.train_one_epoch_m(dataloader,epoch=epoch)
            self.test_m(epoch=epoch)
            print(f'Best test results A: {self.best_test_f1_m[0]:.4f}')
            print(f'Best test results B: {self.best_test_f1_m[1]:.4f}')
            print(f'Best test results C: {self.best_test_f1_m[2]:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1_m, self.best_test_f1_m),
            f'C:/Users/yuval/PycharmProjects/HW3_advanced_ml/Final_project/multi_task_offensive_language_detection/save/results//mtl_{self.datetimestr}_{self.best_test_f1_m[0]:.4f}.pt'
        )
    def data_balancer(self,label_A, label_B):
        ### Balance class a and B
        class_0_a = torch.where(label_A == 0)
        class_1_a = torch.where(label_A == 1)
        if len(class_1_a[0]) ==0 or len(class_0_a[0]) == 0:
            idx_a = [0 ,1]
        else:
            min_len = np.min([len(class_0_a[0]),len(class_1_a[0])])
            index_to_train_a = torch.randperm(min_len)
            idx_a = torch.cat((class_0_a[0][index_to_train_a] , class_1_a[0][index_to_train_a]))

        class_0_b = torch.where(label_B == 0)
        class_1_b = torch.where(label_B == 1)
        if len(class_1_b[0]) ==0 or len(class_0_b[0]) ==0:
            idx_b = [0, 1]
        else:
            min_len = np.min([len(class_0_b[0]),len(class_1_b[0])])
            index_to_train_b = torch.randperm(min_len)
            idx_b = torch.cat((class_0_b[0][index_to_train_b] , class_1_b[0][index_to_train_b]))
        return idx_a, idx_b

    def nan_checker(self, loss):
        zero = torch.tensor(0,device=self.device, dtype=torch.long)
        if torch.isnan(loss):
            return zero
        else:
            return loss

    def train_one_epoch_m(self, dataloader,epoch=None):
        self.model.train()

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None

        _loss_a = torch.tensor(0,device='cuda', dtype=torch.long)
        _loss_b = torch.tensor(0,device='cuda', dtype=torch.long)
        _loss_c = torch.tensor(0,device='cuda', dtype=torch.long)
        iters_per_epoch = 0
        loss_task_a = 0
        loss_task_b = 0
        loss_task_c = 0
        final_loss  = 0
        for inputs, lens, mask, label_A, label_B, label_C, sentence_embedding, b_importance,c_importance  in tqdm(dataloader, desc='Training M'):
            iters_per_epoch += 1
            b_importance = b_importance.to(device=self.device)
            c_importance = c_importance.to(device=self.device)

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device) #* b_importance + 0.0
            label_C = label_C.to(device=self.device) #* c_importance + 0.0
            sentence_embedding = sentence_embedding.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                # logits_A, logits_B, logits_C = self.model(inputs, mask)
                all_logits = self.model(inputs, lens, mask, sentence_embedding)
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1][:, 0:2].argmax(dim=1)
                y_pred_C = all_logits[2][:, 0:3].argmax(dim=1)

                Non_null_index_B = label_B != LABEL_DICT['b']['NULL']
                Non_null_label_B = label_B[Non_null_index_B]
                Non_null_pred_B = y_pred_B[Non_null_index_B]

                Non_null_index_C = label_C != LABEL_DICT['c']['NULL']
                Non_null_label_C = label_C[Non_null_index_C]
                Non_null_pred_C = y_pred_C[Non_null_index_C]

                labels_all_A = label_A.cpu().numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.cpu().numpy()))
                labels_all_B = Non_null_label_B.cpu().numpy() if labels_all_B is None else np.concatenate((labels_all_B, Non_null_label_B.cpu().numpy()))
                labels_all_C = Non_null_label_C.cpu().numpy() if labels_all_C is None else np.concatenate((labels_all_C, Non_null_label_C.cpu().numpy()))

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = Non_null_pred_B.cpu().numpy() if y_pred_all_B is None else np.concatenate((y_pred_all_B, Non_null_pred_B.cpu().numpy()))
                y_pred_all_C = Non_null_pred_C.cpu().numpy() if y_pred_all_C is None else np.concatenate((y_pred_all_C, Non_null_pred_C.cpu().numpy()))


                a_all_logits = all_logits[0]
                b_all_logits_non_null = all_logits[1][Non_null_index_B]
                c_all_logits_non_null = all_logits[2][Non_null_index_C]

                if np.random.rand() > 1.1:
                    idx_a, idx_b = self.data_balancer(label_A,Non_null_label_B )
                    _loss_a = self.loss_weights[0] * torch.mean(self.criterion(a_all_logits[idx_a], label_A[idx_a]))
                    #print(b_all_logits_non_null)
                    if len(b_all_logits_non_null) > 1:
                        _loss_b = self.loss_weights[1] * torch.mean(self.criterion(b_all_logits_non_null[idx_b], torch.tensor(Non_null_label_B[idx_b],dtype=torch.long) )  )
                    else:
                        _loss_b = self.loss_weights[1] * torch.mean(self.criterion(b_all_logits_non_null, torch.tensor(Non_null_label_B,dtype=torch.long) )  )
                else:
                    _loss_a = self.loss_weights[0] * torch.mean(self.criterion(a_all_logits, label_A))
                    _loss_b = self.loss_weights[1] * torch.mean(self.criterion(b_all_logits_non_null, torch.tensor(Non_null_label_B,dtype=torch.long) )  )
                if np.isnan(_loss_a.item()):
                    1

                loss_task_a += np.where(np.isnan(_loss_a.item()),0,_loss_a.item())
                loss_task_b += np.where(np.isnan(_loss_b.item()),0,_loss_b.item())
                _loss_c = self.loss_weights[2] * torch.mean( self.criterion(c_all_logits_non_null , torch.tensor(Non_null_label_C,dtype=torch.long)))
                loss_task_c += np.where(np.isnan(_loss_c.item()),0,_loss_c.item())

                _loss_a = self.nan_checker(_loss_a)
                _loss_b = self.nan_checker(_loss_b)
                _loss_c = self.nan_checker(_loss_c)

                loss = (_loss_a + _loss_b + _loss_c )
                # Backward
                loss = self.nan_checker(loss)
                loss.backward()
                final_loss += loss.item()
                if np.isnan(final_loss):
                    print(loss)
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        wandb.log({"total_loss train epoch": loss_task_a + loss_task_b + loss_task_c},step=epoch)
        wandb.log({"loss task a  epoch": loss_task_a},step=epoch)
        wandb.log({"loss task b  epoch": loss_task_b},step=epoch)
        wandb.log({"loss task c  epoch": loss_task_c},step=epoch)

        wandb.watch(self.model)

        final_loss /= iters_per_epoch
        f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')

        print(f'loss = {final_loss:.4f}')
        print(f'A: {f1_A:.4f}')
        print(f'B: {f1_B:.4f}')
        print(f'C: {f1_C:.4f}')

        self.train_losses.append(final_loss)
        self.train_f1.append([f1_A, f1_B, f1_C])

        if f1_A > self.best_train_f1_m[0]:
            self.best_train_f1_m[0] = f1_A
        if f1_B > self.best_train_f1_m[1]:
            self.best_train_f1_m[1] = f1_B
        if f1_C > self.best_train_f1_m[2]:
            self.best_train_f1_m[2] = f1_C

    def test_m(self, epoch = None):
        self.model.eval()
        dataloader = self.dataloaders['test']
        loss = 0
        iters_per_epoch = 0

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None
        loss_a = 0
        loss_b = 0
        loss_c = 0
        for inputs, lens, mask, label_A, label_B, label_C, sentence_embedding, b_importance,c_importance  in tqdm(dataloader, desc='Test M'):
            iters_per_epoch += 1

            b_importance = b_importance.to(device=self.device)
            c_importance = c_importance.to(device=self.device)
            sentence_embedding = sentence_embedding.to(device=self.device)
            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device) * b_importance
            label_C = label_C.to(device=self.device) * c_importance

            with torch.set_grad_enabled(False):
                #padded_input = torch.zeros(inputs.shape[0], 125, device=self.device,dtype=torch.long)
                #padded_mask = torch.zeros(inputs.shape[0], 125, device=self.device,dtype=torch.long)
#
                #padded_input[:,0:inputs.shape[1]] = inputs
                #padded_mask[:,0:inputs.shape[1]] = mask

                all_logits = self.model(inputs, lens, mask,sentence_embedding)

                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1].argmax(dim=1).cpu().numpy()
                y_pred_C = all_logits[2].argmax(dim=1).cpu().numpy()

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(label_B, y_pred_B)
                # f1[2] += self.calc_f1(label_C, y_pred_C)
                Non_null_index_B = (label_B != LABEL_DICT['b']['NULL'] ).cpu()
                Non_null_label_B = label_B[Non_null_index_B].cpu()
                Non_null_pred_B = y_pred_B[Non_null_index_B]

                Non_null_index_C = (label_C != LABEL_DICT['c']['NULL']).cpu()
                Non_null_label_C = label_C[Non_null_index_C].cpu()
                Non_null_pred_C = y_pred_C[Non_null_index_C]

               #y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
               #y_pred_all_B = y_pred_B if y_pred_all_B is None else np.concatenate((y_pred_all_B, y_pred_B*b_importance.cpu().numpy()))
               #y_pred_all_C = y_pred_C if y_pred_all_C is None else np.concatenate((y_pred_all_C, y_pred_C*c_importance.cpu().numpy()))
                labels_all_A = label_A.cpu().numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.cpu().numpy()))
                labels_all_B = Non_null_label_B.numpy() if labels_all_B is None else np.concatenate((labels_all_B, Non_null_label_B.numpy()))
                labels_all_C = Non_null_label_C.numpy() if labels_all_C is None else np.concatenate((labels_all_C, Non_null_label_C.numpy()))

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = Non_null_pred_B if y_pred_all_B is None else np.concatenate((y_pred_all_B, Non_null_pred_B))
                y_pred_all_C = Non_null_pred_C if y_pred_all_C is None else np.concatenate((y_pred_all_C, Non_null_pred_C))

                _loss_a = self.loss_weights[0] * torch.mean(self.criterion(all_logits[0], label_A))
                loss_a += _loss_a.item()
                _loss_b = self.loss_weights[1] * torch.mean(self.criterion(all_logits[1], torch.tensor(label_B,dtype=torch.long) ) * b_importance)
                loss_b += _loss_b.item()
                _loss_c = self.loss_weights[2] * torch.mean(self.criterion(all_logits[2], torch.tensor(label_C,dtype=torch.long) ) * c_importance)
                loss_c += _loss_c.item()

                _loss = _loss_a + _loss_c + _loss_b
                loss += _loss.item()

        loss /= iters_per_epoch
        f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')

        wandb.log({"Test total_loss": loss},step=epoch)
        wandb.log({"Test loss task a": loss_a },step=epoch)
        wandb.log({"Test loss task b": loss_b },step=epoch)
        wandb.log({"Test loss task c": loss_c },step=epoch)

        print(f'loss = {loss:.4f}')
        print(f'A: {f1_A:.4f}')
        print(f'B: {f1_B:.4f}')
        print(f'C: {f1_C:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append([f1_A, f1_B, f1_C])

        if f1_A > self.best_test_f1_m[0]:
            self.best_test_f1_m[0] = f1_A
            self.save_model()
        if f1_B > self.best_test_f1_m[1]:
            self.best_test_f1_m[1] = f1_B
        if f1_C > self.best_test_f1_m[2]:
            self.best_test_f1_m[2] = f1_C

        # for i in range(len(f1)):
        #     for j in range(len(f1[0])):
        #         if f1[i][j] > self.best_test_f1_m[i][j]:
        #             self.best_test_f1_m[i][j] = f1[i][j]
        #             if i == 0 and j == 0:
        #                 self.save_model()

    def calc_f1(self, labels, y_pred):
        return np.array([
            f1_score(labels.cpu(), y_pred.cpu(), average='macro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='micro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='weighted')
        ], np.float64)

    def printing(self, loss, f1):
        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1[0]:.4f}')
        # print(f'Micro-F1 = {f1[1]:.4f}')
        # print(f'Weighted-F1 = {f1[2]:.4f}')

    def save_model(self):
        print('Saving model...')
        if self.task_name == 'all':
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1_m[0]}_seed{self.seed}.pt'
        else:
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1}_seed{self.seed}.pt'
        save(copy.deepcopy(self.model.state_dict()), filename)
