import os
import numpy as np
from torch.utils.data import DataLoader
from multi_task_offensive_language_detection.data import task_a, task_b, task_c, all_tasks, read_test_file, read_test_file_all
from multi_task_offensive_language_detection.config import OLID_PATH
from multi_task_offensive_language_detection.cli import get_args
from multi_task_offensive_language_detection.utils import load
from multi_task_offensive_language_detection.datasets import HuggingfaceDataset, HuggingfaceMTDataset, ImbalancedDatasetSampler
from multi_task_offensive_language_detection.models.bert import BERT, RoBERTa
from multi_task_offensive_language_detection.models.gated import GatedModel
from multi_task_offensive_language_detection.models.project_models import ODF, ODF2, MTL_Transformer_LSTM2, Paper_recon
from transformers import BertTokenizer, RobertaTokenizer, get_cosine_schedule_with_warmup
from multi_task_offensive_language_detection.trainer import Trainer
from transformers import XLMTokenizer
from sentence_transformers import SentenceTransformer
import wandb
TRAIN_PATH = os.path.join(OLID_PATH, 'olid-training-v1.0.tsv')
import torch
if __name__ == '__main__':
   #############################################################################################
    #### Training and network configurations

    wandb.init(project="my-test-project")
    args = get_args()
    task = args['task']
    model_name = args['model']
    model_size = args['model_size']
    truncate = args['truncate']
    epochs = args['epochs']
    lr = args['learning_rate']
    wd = args['weight_decay']
    bs = args['batch_size']
    patience = args['patience']
    seed = args['seed']
   ##############################################################################################
    torch.manual_seed(seed)
    np.random.seed(seed)
    sentence_transformer =  SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_labels = 3 if task == 'c' else 2

    # Choose the customize tokenizer for each pre-trained model
    if model_name == 'xlm_r':
        if task == 'all':
            model = Paper_recon(model_name, model_size, args=args)
        tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-ende-1024")
        #tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    if model_name == 'bert':
            if task == 'all':
                model = ODF(model_name, model_size, args=args)

            #xlmr
                #model = ODF(model_name, model_size, args=args)

            else:
                model = BERT(model_size, args=args, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        if task == 'all':
            model = ODF(model_name, model_size, args=args)
        else:
            model = RoBERTa(model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
    elif model_name == 'bert-gate' and task == 'all':
        model_name = model_name.replace('-gate', '')
        model = GatedModel(model_name, model_size, args=args)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta-gate' and task == 'all':
        model_name = model_name.replace('-gate', '')
        model = GatedModel(model_name, model_size, args=args)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')

    # Move model to correct device
    model = model.to(device=device)

    if args['ckpt'] != '':
        model.load_state_dict(load(args['ckpt']),strict=False)

    # Read in data depends on different subtasks
    # label_orders = {'a': ['OFF', 'NOT'], 'b': ['TIN', 'UNT'], 'c': ['IND', 'GRP', 'OTH']}
    if task in ['a', 'b', 'c']:
        data_methods = {'a': task_a, 'b': task_b, 'c': task_c}
        ids, token_ids, lens, mask, labels = data_methods[task](TRAIN_PATH, tokenizer=tokenizer, truncate=truncate)
        test_ids, test_token_ids, test_lens, test_mask, test_labels = read_test_file(task, tokenizer=tokenizer, truncate=truncate, sentence_transformer=sentence_transformer)
        _Dataset = HuggingfaceDataset
    elif task in ['all']:
        ids, token_ids, lens, mask, label_a, label_b, label_c, sentence_embedding = all_tasks(TRAIN_PATH, tokenizer=tokenizer, truncate=truncate, sentence_transformer=sentence_transformer)
        test_ids, test_token_ids, test_lens, test_mask, test_label_a, test_label_b, test_label_c, test_sentence_embedding = read_test_file_all(tokenizer, sentence_transformer=sentence_transformer)
        labels = {'a': label_a, 'b': label_b, 'c': label_c}
        test_labels = {'a': test_label_a, 'b': test_label_b, 'c': test_label_c}
        _Dataset = HuggingfaceMTDataset

    datasets = {
        'train': _Dataset(
            sentence_embedding = sentence_embedding,
            model= model,
            input_ids=token_ids,
            lens=lens,
            mask=mask,
            labels=labels,
            task=task
        ),
        'test': _Dataset(
            sentence_embedding = test_sentence_embedding,
            input_ids=test_token_ids,
            lens=test_lens,
            mask=test_mask,
            labels=test_labels,
            task=task
        )
    }

    sampler = ImbalancedDatasetSampler(datasets['train']) if task in ['a', 'b', 'c'] else None
    dataloaders = {
        'train': DataLoader(
            dataset=datasets['train'],
            batch_size=bs,
            sampler=sampler
        ),
        'test': DataLoader(dataset=datasets['test'], batch_size=bs)
    }

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if args['scheduler']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        # A warmup scheduler
        t_total = epochs * len(dataloaders['train'])
        warmup_steps = np.ceil(t_total / 10.0) * 2
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = None

    trainer = Trainer(
        model=model,
        epochs=epochs,
        dataloaders=dataloaders,
        criterion=criterion,
        loss_weights=args['loss_weights'],
        clip=args['clip'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        task_name=task,
        model_name=model_name,
        seed=args['seed']
    )

    if task in ['a', 'b', 'c']:
        trainer.train()
    else:
        trainer.train_m()
