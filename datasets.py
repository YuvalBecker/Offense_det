import torch
from torch.utils.data import Dataset
from multi_task_offensive_language_detection.config import LABEL_DICT
import numpy as np
from collections import defaultdict
class HuggingfaceDataset(Dataset):
    def __init__(self, input_ids, lens, mask, labels, task):
        self.input_ids = torch.tensor(input_ids)
        self.lens = lens
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels
        self.task = task

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        this_LABEL_DICT = LABEL_DICT[self.task]
        input = self.input_ids[idx]
        length = self.lens[idx]
        mask = self.mask[idx]
        label = torch.tensor(this_LABEL_DICT[self.labels[idx]])
        return input, length, mask, label

class HuggingfaceMTDataset(Dataset):
    def __init__(self, input_ids=None, lens=None, mask=None, labels=None, task=None, sentence_embedding=None, model=None):
        self.input_ids = torch.tensor(input_ids)
        self.lens = lens
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels
        self.sentence_embedding = sentence_embedding
        self.model = model
        self.training_list = np.arange(0, self.labels['a'].shape[0] , 1)
        self.device = 'cuda'
        #if model != None:
        #    self.hard_collection(percent_hard=25)
    #def __len__(self):
    #    return self.labels['a'].shape[0]
    def __len__(self):
        return self.training_list.shape[0]

    def store_error_per_class(self, chosen_dict, error_vlaues, labels, indexes ):
        for (err, label, index) in zip(error_vlaues, labels, indexes):
            chosen_dict[str(label.cpu().detach().numpy())]['values'] += [err]
            chosen_dict[str(label.cpu().detach().numpy())]['index'] += [index]
    def hard_collection(self, percent_hard = 20):
        error_list_a = []
        error_list_b = []
        error_list_c = []
        dict_errors = defaultdict(dict)
        dict_errors['a']['0'] = defaultdict(list)
        dict_errors['a']['1'] = defaultdict(list)
        dict_errors['a']['1'] = defaultdict(list)

        dict_errors['b']['0'] = defaultdict(list)
        dict_errors['b']['1'] = defaultdict(list)

        dict_errors['c']['0'] = defaultdict(list)
        dict_errors['c']['1'] = defaultdict(list)
        dict_errors['c']['2'] = defaultdict(list)


        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        batch_size = 14
        for batch_id in range(int(  ( self.labels['a'].shape[0] )  / batch_size )):
            input = self.input_ids[batch_id * batch_size :batch_id * batch_size + batch_size ]
            ids_samples_a = list(range(batch_id * batch_size,batch_id * batch_size + batch_size ))
            ids_samples_b = list(range(batch_id * batch_size,batch_id * batch_size + batch_size ))
            ids_samples_c = list(range(batch_id * batch_size,batch_id * batch_size + batch_size ))

            sentence_embedding = self.sentence_embedding[batch_id * batch_size :batch_id * batch_size + batch_size ]
            mask =  self.mask[batch_id * batch_size :batch_id * batch_size + batch_size ]
            length = self.lens[batch_id * batch_size :batch_id * batch_size + batch_size ]

            label_A = [LABEL_DICT['a'][self.labels['a'][batch_id * batch_size + idx]] for idx in range(batch_size)]
            label_B = [LABEL_DICT['b'][self.labels['b'][batch_id * batch_size + idx]] for idx in range(batch_size)]
            label_C = [LABEL_DICT['c'][self.labels['c'][batch_id * batch_size + idx]] for idx in range(batch_size)]

            #label_A = torch.tensor(LABEL_DICT['a'][self.labels['a'][batch_id * batch_size :batch_id * batch_size + batch_size]])
            #label_B = torch.tensor(LABEL_DICT['b'][self.labels['b'][idx]])
            #label_C = torch.tensor(LABEL_DICT['c'][self.labels['c'][idx]])

            input = input.to(device=self.device)
            lens = torch.tensor(length).to(device=self.device)
            mask = torch.tensor(mask).to(device=self.device)
            label_A = torch.tensor(label_A,device=self.device)
            label_B = torch.tensor(label_B,device=self.device) #* b_importance + 0.0
            label_C = torch.tensor(label_C,device=self.device)# * c_importance + 0.0
            sentence_embedding = torch.tensor(sentence_embedding).to(device=self.device)

                # Forward
                # logits_A, logits_B, logits_C = self.model(inputs, mask)
            all_logits = self.model(input, lens, mask, sentence_embedding)
            Non_null_index_B = label_B != LABEL_DICT['b']['NULL']
            idx_b = np.where(Non_null_index_B.cpu().detach().numpy())[0]
            ids_samples_b = [ids_samples_b[idb] for idb in idx_b]

            Non_null_label_B = label_B[Non_null_index_B]

            Non_null_index_C = label_C != LABEL_DICT['c']['NULL']
            idx_c = np.where(Non_null_index_C.cpu().detach().numpy())[0]
            ids_samples_c = [ids_samples_c[idc] for idc in idx_c]

            Non_null_label_C = label_C[Non_null_index_C]

            # f1[0] += self.calc_f1(label_A, y_pred_A)
            # f1[1] += self.calc_f1(Non_null_label_B, Non_null_pred_B)
            # f1[2] += self.calc_f1(Non_null_label_C, Non_null_pred_C)
            a_all_logits = all_logits[0]
            b_all_logits_non_null = all_logits[1][Non_null_index_B]
            c_all_logits_non_null = all_logits[2][Non_null_index_C]


            _loss_a = list(1 * criterion(a_all_logits, label_A).cpu().detach().numpy())
            _loss_b = list(1 * criterion(b_all_logits_non_null, torch.tensor(Non_null_label_B,dtype=torch.long) ).cpu().detach().numpy())
            _loss_c = list(1 *  criterion(c_all_logits_non_null , torch.tensor(Non_null_label_C,dtype=torch.long)).cpu().detach().numpy())
            _loss_a= _loss_a * (1 * ~np.isnan(_loss_a) )
            _loss_b= _loss_b * (1 * ~np.isnan(_loss_b) )
            _loss_c= _loss_c * (1 * ~np.isnan(_loss_c) )


            self.store_error_per_class(dict_errors['a'], _loss_a, label_A, ids_samples_a )
            self.store_error_per_class(dict_errors['b'], _loss_b, Non_null_label_B,ids_samples_b )
            self.store_error_per_class(dict_errors['c'], _loss_b, Non_null_label_C ,ids_samples_c)

            error_list_a+=list(_loss_a)
            error_list_b+=list(_loss_b)
            error_list_c+=list(_loss_c)
        #combined_list = dict_errors['b']['1']['index'] + dict_errors['c']['2']['index'] + dict_errors['a']['1']['index']

        sorted_args_a_0 = [dict_errors['a']['0']['index'][id] for id in list(np.argsort(dict_errors['a']['0']['values'])[::-1][0:int(len(dict_errors['a']['0']['values']) * ( 1 * percent_hard  / 100) ) ])]
        sorted_args_a_1 = [dict_errors['a']['1']['index'][id] for id in list(np.argsort(dict_errors['a']['1']['values'])[::-1][0:int(len(dict_errors['a']['1']['values']) * ( 0.5 * percent_hard  / 100) ) ])]

        sorted_args_b_0 = [dict_errors['b']['0']['index'][id] for id in list(np.argsort(dict_errors['b']['0']['values'])[::-1][0:int(len(dict_errors['b']['0']['values']) * ( 0.5 * percent_hard  / 100) ) ])]
        sorted_args_b_1 = [dict_errors['b']['1']['index'][id] for id in list(np.argsort(dict_errors['b']['1']['values'])[::-1][0:int(len(dict_errors['b']['1']['values']) * ( 3 * percent_hard  / 100) ) ])]

        sorted_args_c_0 = [dict_errors['c']['0']['index'][id] for id in list(np.argsort(dict_errors['c']['0']['values'])[::-1][0:int(len(dict_errors['c']['0']['values'])  * ( 0.5 * percent_hard  / 100) ) ])]
        sorted_args_c_1 = [dict_errors['c']['1']['index'][id] for id in list(np.argsort(dict_errors['c']['1']['values'])[::-1][0:int(len(dict_errors['c']['1']['values']) * ( 0.5 * percent_hard  / 100) ) ])]
        sorted_args_c_2 = [dict_errors['c']['2']['index'][id] for id in list(np.argsort(dict_errors['c']['2']['values'])[::-1][0:int(len(dict_errors['c']['2']['values']) * ( 1 * percent_hard  / 100) ) ])]

        combined_list = sorted_args_a_0 + sorted_args_a_1  + sorted_args_b_0 + sorted_args_b_1 + sorted_args_c_0 + sorted_args_c_1 + sorted_args_c_2
        self.training_list =  np.array(combined_list)

    def __getitem__(self, idx):
        idx = self.training_list[idx]
        input = self.input_ids[idx]
        sentence_embedding = self.sentence_embedding[idx]
        mask =  self.mask[idx]
        length = self.lens[idx]
        label_A = torch.tensor(LABEL_DICT['a'][self.labels['a'][idx]])
        label_B = torch.tensor(LABEL_DICT['b'][self.labels['b'][idx]])
        label_C = torch.tensor(LABEL_DICT['c'][self.labels['c'][idx]])
        b_importance = torch.tensor(0.0 + int(label_A == 0) )
        c_importance = torch.tensor(0.0 + int(label_B == 0) )

        return input, length, mask, label_A, label_B, label_C, sentence_embedding, b_importance, c_importance

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset.labels))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, id_):
        return dataset.labels[id_]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
