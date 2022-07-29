import torch
from torch import nn
from transformers import BertModel, RobertaModel
from .modules.attention import Attention
from torch.nn import MultiheadAttention
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification
xlmr = torch.hub.load(r'pytorch/fairseq', r'xlmr.base')
xlmr.requires_grad_(False)
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import XLMModel

# Original class
class ODF(nn.Module):

    def __init__(self, model, model_size, args):
        super(ODF, self).__init__()
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768 if model_size == 'base' else 1024
        self.with_sentence_emb = True

        if model == 'bert':
            MODEL = BertModel
            model_full_name = f'{model}-{model_size}-uncased'
            self.emb = MODEL.from_pretrained(
                model_full_name,
                hidden_dropout_prob=args['hidden_dropout'],
                attention_probs_dropout_prob=args['attention_dropout']
            )
        elif model == 'roberta':
            MODEL = RobertaModel
            model_full_name = f'{model}-{model_size}'

        elif model == 'xlm_r':
            MODEL = AutoModelForMaskedLM
            model_full_name = "xlm-roberta-base"
            self.emb = torch.hub.load(r'pytorch/fairseq', r'xlmr.base')


        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'b': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'c': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })

        self.attention_layers = nn.ModuleDict({
            'a': Attention(hidden_size * 2),
            'b': Attention(hidden_size * 2),
            'c': Attention(hidden_size * 2)
        })

        self.dropout = nn.Dropout(p=args['dropout'])

        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(600, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2), nn.Softmax(dim=1),
            ),
        })

        if self.with_sentence_emb:
            self.sentence_emb =  nn.Sequential(
                nn.Linear(384 , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 20 ))

        self.b = nn.Sequential(
            nn.Linear(600 + 2 + self.with_sentence_emb*20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2), nn.Softmax(dim=1))

        self.c = nn.Sequential(
            nn.Linear(600 + 2+ self.with_sentence_emb*20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3), nn.Softmax(dim=1))

        self.Linears_key_value_query_a = nn.ModuleDict({
            'key': nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'query':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'value':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
        })
        self.Linears_key_value_query_b =   nn.ModuleDict({
            'key': nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'query':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'value':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
        })
        self.Linears_key_value_query_c =  nn.ModuleDict({
            'key': nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'query':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'value':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
        })

        self.attn_dim_match_a = nn.Sequential(nn.Linear(30, 600),nn.ReLU())
        self.attn_dim_match_b = nn.Sequential(nn.Linear(30, 600),nn.ReLU())
        self.attn_dim_match_c = nn.Sequential(nn.Linear(30, 600),nn.ReLU())

        self.MHA_a = MultiheadAttention(embed_dim=30, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.MHA_b = MultiheadAttention(embed_dim=30, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.MHA_c = MultiheadAttention(embed_dim=30, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)

    def forward(self, inputs, lens, mask, sentence_embedding):
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)
        # Sequence embedding per time step
        #oo = self.xlmr.extract_features(inputs, return_all_hiddens=False)
        _, (h_a, _) = self.LSTMs['a'](embs)
        if self.concat:
            h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        else:
            h_a = h_a[0] + h_a[1]
        h_a = self.dropout(h_a)

        _, (h_b, _) = self.LSTMs['b'](embs)
        if self.concat:
            h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        else:
            h_b = h_b[0] + h_b[1]
        h_b = self.dropout(h_b)

        _, (h_c, _) = self.LSTMs['c'](embs)
        if self.concat:
            h_c = torch.cat((h_c[0], h_c[1]), dim=1)
        else:
            h_c = h_c[0] + h_c[1]
        h_c = self.dropout(h_c)
        if self.with_sentence_emb:
            out_emb_features = self.sentence_emb(sentence_embedding)


        key_a = self.Linears_key_value_query_a['key'](embs)
        query_a = self.Linears_key_value_query_a['query'](embs)
        value_a = self.Linears_key_value_query_a['value'](embs)

        key_b = self.Linears_key_value_query_b['key'](embs)
        query_b = self.Linears_key_value_query_b['query'](embs)
        value_b = self.Linears_key_value_query_b['value'](embs)

        key_c = self.Linears_key_value_query_c['key'](embs)
        query_c = self.Linears_key_value_query_c['query'](embs)
        value_c = self.Linears_key_value_query_c['value'](embs)

        out_attn_a, attn_weights = self.MHA_a(key = key_a, query=query_a, value=value_a)
        out_attn_b, attn_weights = self.MHA_b(key = key_b, query=query_b, value=value_b)
        out_attn_c, attn_weights = self.MHA_c(key = key_c, query=query_c, value=value_c)

        out_attn_a = torch.nn.Flatten()(torch.sum(out_attn_a, dim=1))
        out_attn_b = torch.nn.Flatten()(torch.sum(out_attn_b, dim=1))
        out_attn_c = torch.nn.Flatten()(torch.sum(out_attn_c, dim=1))

        out_attn_a = self.attn_dim_match_a(out_attn_a)
        out_attn_b = self.attn_dim_match_a(out_attn_b)
        out_attn_c = self.attn_dim_match_a(out_attn_c)

        logits_a = self.Linears['a'](out_attn_a * h_a)
        logits_b = self.b(torch.cat([logits_a.detach(), out_attn_b * h_b, out_emb_features],dim=1))
        logits_c = self.c(torch.cat([logits_a.detach(), out_attn_c * h_c, out_emb_features],dim=1))

        return logits_a, logits_b, logits_c

class Paper_recon(nn.Module):
    '''
    Training procedure and model according to the anchor paper.
    '''
    def __init__(self, model, model_size, args):
        super(Paper_recon, self).__init__()
        self.concat = args['hidden_combine_method'] == 'concat'
        self.with_sentence_emb = True
        self.emb = XLMModel.from_pretrained("xlm-mlm-ende-1024")


        self.a = nn.Sequential(
            nn.Linear(1024, 50),
            nn.ReLU(),
            nn.Linear(50, 2), nn.Softmax(dim=1)).cuda()
        self.b = nn.Sequential(
            nn.Linear(1024, 50),
            nn.ReLU(),
            nn.Linear(50, 2), nn.Softmax(dim=1)).cuda()
        self.c = nn.Sequential(
            nn.Linear(1024, 50),
            nn.ReLU(),
            nn.Linear(50, 3), nn.Softmax(dim=1)).cuda()

    def forward(self, inputs, lens, mask, sentence_embedding):
        embs = self.emb(inputs)[0] # (batch_size, sequence_length, hidden_size)
        embs_summed = torch.sum(embs,dim=[1])
        logits_a = self.a(embs_summed)
        logits_b = self.b(embs_summed)
        logits_c = self.c(embs_summed)
        return logits_a, logits_b, logits_c

class ODF2(nn.Module):
    def __init__(self, model, model_size, args):
        super(ODF2, self).__init__()
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768 if model_size == 'base' else 768
        self.with_sentence_emb = True
        self.xlmr = torch.hub.load(r'pytorch/fairseq', r'xlmr.base')
        self.xlmr.requires_grad_(False)

        if model == 'bert':
                    MODEL = BertModel
                    model_full_name = f'{model}-{model_size}-uncased'
        elif model == 'roberta':
                    MODEL = RobertaModel
                    model_full_name = f'{model}-{model_size}'

        self.emb = MODEL.from_pretrained(
            model_full_name,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'b': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'c': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })

        self.attention_layers = nn.ModuleDict({
            'a': Attention(hidden_size * 2),
            'b': Attention(hidden_size * 2),
            'c': Attention(hidden_size * 2)
        })

        self.dropout = nn.Dropout(p=args['dropout'])

        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(600, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2), nn.Softmax(dim=1),
            ),
        })

        if self.with_sentence_emb:
            self.sentence_emb =  nn.Sequential(
                nn.Linear(384 , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 20 ))

        self.b = nn.Sequential(
            nn.Linear(600 + 2 + self.with_sentence_emb*20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2), nn.Softmax(dim=1))

        self.c = nn.Sequential(
            nn.Linear(600 + 2+ self.with_sentence_emb*20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3), nn.Softmax(dim=1))

        self.Linears_key_value_query_a = nn.ModuleDict({
            'key': nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'query':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'value':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
        })
        self.Linears_key_value_query_b =   nn.ModuleDict({
            'key': nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'query':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'value':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
        })
        self.Linears_key_value_query_c =  nn.ModuleDict({
            'key': nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'query':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
            'value':nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 30),nn.ReLU()),
        })

        self.attn_dim_match_a = nn.Sequential(nn.Linear(30, 600),nn.ReLU())
        self.attn_dim_match_b = nn.Sequential(nn.Linear(30, 600),nn.ReLU())
        self.attn_dim_match_c = nn.Sequential(nn.Linear(30, 600),nn.ReLU())

        self.MHA_a = MultiheadAttention(embed_dim=30, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.MHA_b = MultiheadAttention(embed_dim=30, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.MHA_c = MultiheadAttention(embed_dim=30, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)

    def forward(self, inputs, lens, mask, sentence_embedding):
        #embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)
        # Sequence embedding per time step
        embs = self.xlmr.extract_features(inputs, return_all_hiddens=False)
        _, (h_a, _) = self.LSTMs['a'](embs)
        if self.concat:
            h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        else:
            h_a = h_a[0] + h_a[1]
        h_a = self.dropout(h_a)

        _, (h_b, _) = self.LSTMs['b'](embs)
        if self.concat:
            h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        else:
            h_b = h_b[0] + h_b[1]
        h_b = self.dropout(h_b)

        _, (h_c, _) = self.LSTMs['c'](embs)
        if self.concat:
            h_c = torch.cat((h_c[0], h_c[1]), dim=1)
        else:
            h_c = h_c[0] + h_c[1]
        h_c = self.dropout(h_c)
        if self.with_sentence_emb:
            out_emb_features = self.sentence_emb(sentence_embedding)


        key_a = self.Linears_key_value_query_a['key'](embs)
        query_a = self.Linears_key_value_query_a['query'](embs)
        value_a = self.Linears_key_value_query_a['value'](embs)

        key_b = self.Linears_key_value_query_b['key'](embs)
        query_b = self.Linears_key_value_query_b['query'](embs)
        value_b = self.Linears_key_value_query_b['value'](embs)

        key_c = self.Linears_key_value_query_c['key'](embs)
        query_c = self.Linears_key_value_query_c['query'](embs)
        value_c = self.Linears_key_value_query_c['value'](embs)

        out_attn_a, attn_weights = self.MHA_a(key = key_a, query=query_a, value=value_a)
        out_attn_b, attn_weights = self.MHA_b(key = key_b, query=query_b, value=value_b)
        out_attn_c, attn_weights = self.MHA_c(key = key_c, query=query_c, value=value_c)

        out_attn_a = torch.nn.Flatten()(torch.sum(out_attn_a, dim=1))
        out_attn_b = torch.nn.Flatten()(torch.sum(out_attn_b, dim=1))
        out_attn_c = torch.nn.Flatten()(torch.sum(out_attn_c, dim=1))

        out_attn_a = self.attn_dim_match_a(out_attn_a)
        out_attn_b = self.attn_dim_match_a(out_attn_b)
        out_attn_c = self.attn_dim_match_a(out_attn_c)

        logits_a = self.Linears['a'](out_attn_a * h_a)
        logits_b = self.b(torch.cat([logits_a.detach(), out_attn_b * h_b, out_emb_features],dim=1))
        logits_c = self.c(torch.cat([logits_a.detach(), out_attn_c * h_c, out_emb_features],dim=1))

        return logits_a, logits_b, logits_c

class MTL_Transformer_LSTM2(nn.Module):
    def __init__(self, model, model_size, args):
        super(MTL_Transformer_LSTM2, self).__init__()
        hidden_size = args['hidden_size']
        hidden_size = 25
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768 if model_size == 'base' else 1024
        ## Need to create a matrix that maps to keys. queries and values .... in here just
        self.MHA_a = MultiheadAttention(embed_dim=15, num_heads=3, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.MHA_b = MultiheadAttention(embed_dim=15, num_heads=3, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.MHA_c = MultiheadAttention(embed_dim=15, num_heads=3, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)

    #self.query = torch.nn.Parameter(torch.randn())
        if model == 'bert':
            MODEL = BertModel
            model_full_name = f'{model}-{model_size}-uncased'
        elif model == 'roberta':
            MODEL = RobertaModel
            model_full_name = f'{model}-{model_size}'

        self.emb = MODEL.from_pretrained(
            model_full_name,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'b': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'c': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })


        self.attention_layers = nn.ModuleDict({
            'a': Attention(hidden_size * 2),
            'b': Attention(hidden_size * 2),
            'c': Attention(hidden_size * 2)
        })

        self.dropout = nn.Dropout(p=args['dropout'])

        linear_in_features = hidden_size * 2 if self.concat else hidden_size
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 25)
            ),
            'b': nn.Sequential(
                nn.Linear(linear_in_features , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 25)
            ),
            'c': nn.Sequential(
                nn.Linear(linear_in_features , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 25)
            ),
            'sentence': nn.Sequential(
                nn.Linear(384 , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 25)
            ),
            'final_a': nn.Sequential(
                nn.Linear(60 , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            ),
            'final_b': nn.Sequential(
                nn.Linear(75 , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3)
            ),
            'final_c': nn.Sequential(
                nn.Linear(75 , hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 4)
            )
        })
        self.key =nn.Sequential(
            nn.Linear(25 , 15),
            nn.ReLU(),
            nn.Linear(15, 15))
        self.value = nn.Sequential(
            nn.Linear(25 , 15),
            nn.ReLU(),
            nn.Linear(15, 15))
        self.query = nn.Sequential(
            nn.Linear(25 , 15),
            nn.ReLU(),
            nn.Linear(15, 15))

        self.Linears_key_value_query_a = nn.ModuleDict({
            'key': nn.Sequential(
                nn.Linear(25, 15),
            ),
            'value': nn.Sequential(
                nn.Linear(25 , 15),
            ),
            'query': nn.Sequential(
                nn.Linear(25 , 15),
            )
        })
        self.Linears_key_value_query_b = nn.ModuleDict({
            'key': nn.Sequential(
                nn.Linear(25, 15),
            ),
            'value': nn.Sequential(
                nn.Linear(25 , 15),
            ),
            'query': nn.Sequential(
                nn.Linear(25 , 15),
            )
        })
        self.Linears_key_value_query_c = nn.ModuleDict({
            'key': nn.Sequential(
                nn.Linear(25, 15),
            ),
            'value': nn.Sequential(
                nn.Linear(25 , 15),
            ),
            'query': nn.Sequential(
                nn.Linear(25 , 15),
            )
        })

    def forward(self, inputs, lens, mask,sentence_embedding):

        #embs = self.emb_per_word(inputs, attention_mask=mask)[0] # (bch_size, sequence_length, hidden_size)
        sentence_embedding = torch.tensor(sentence_embedding, device='cuda')
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)
        _, (h_a, _) = self.LSTMs['a'](embs)
        if self.concat:
            h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        else:
            h_a = h_a[0] + h_a[1]
        h_a = self.dropout(h_a)
#
        _, (h_b, _) = self.LSTMs['b'](embs)
        if self.concat:
            h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        else:
            h_b = h_b[0] + h_b[1]
        h_b = self.dropout(h_b)
#
        _, (h_c, _) = self.LSTMs['c'](embs)
        if self.concat:
            h_c = torch.cat((h_c[0], h_c[1]), dim=1)
        else:
            h_c = h_c[0] + h_c[1]
        h_c = self.dropout(h_c)

        embedd_task_a = self.Linears['a'](h_a).unsqueeze(1)
        embedd_task_b = self.Linears['b'](h_b).unsqueeze(1)
        embedd_task_c = self.Linears['c'](h_c).unsqueeze(1)

        embedd_sentence = self.Linears['sentence'](sentence_embedding)

        # Weighted features

        concat_multi_task_a = torch.cat([embedd_sentence.unsqueeze(1), embedd_task_a,embedd_task_b,embedd_task_c], dim=1)
        key_a = self.Linears_key_value_query_a['key'](concat_multi_task_a )
        value_a =  self.Linears_key_value_query_a['query'](concat_multi_task_a )
        query_a = self.Linears_key_value_query_a['value'](concat_multi_task_a )

        logits_a = self.Linears['final_a'](torch.nn.Flatten()(self.MHA_a(key=key_a,query=query_a,value=value_a)[0]))
        logits_a_sequence = torch.repeat_interleave(logits_a.unsqueeze(2), repeats=25,dim=2)
        concat_multi_task_bc = torch.cat([embedd_sentence.unsqueeze(1), logits_a_sequence,embedd_task_b,embedd_task_c], dim=1)
        key_b = self.Linears_key_value_query_b['key'](concat_multi_task_bc )
        value_b =  self.Linears_key_value_query_b['query'](concat_multi_task_bc )
        query_b = self.Linears_key_value_query_b['value'](concat_multi_task_bc )

        logits_b = self.Linears['final_b'](torch.nn.Flatten()(self.MHA_b(key=key_b,query=query_b,value=value_b)[0]))

        key_c = self.Linears_key_value_query_c['key'](concat_multi_task_bc )
        value_c =  self.Linears_key_value_query_c['query'](concat_multi_task_bc )
        query_c = self.Linears_key_value_query_c['value'](concat_multi_task_bc )

        logits_c = self.Linears['final_c'](torch.nn.Flatten()(self.MHA_c(key=key_c,query=query_c,value=value_c)[0]))

        return logits_a, logits_b, logits_c
