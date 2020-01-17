import os
import pickle
from typing import Union

import numpy

import torch
import torch.nn as nn
import torch.nn.functional
from torch.nn.modules.loss import _Loss

from deepreg.datasets.calls_dataset import CallsDataset_V2
from deepreg.tools.config import SPECIAL_TOKEN_IDS, PAD, BOD
from deepreg.tools.sequential import Sequential



class FinanceModel(nn.Module):

    def __init__(self,
                 train_dataset,
                 hidden_size=300,
                 hidden_layers=3,
                 dropout=0.2,
                 sparse=False,
                 init_std=False,
                 make_ff=True,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True
                 ):
        super().__init__()
        in_size = train_dataset.data_stats['feature_size']
        if make_ff:
            self.ff = self._make_ff(dropout, in_size, hidden_size, hidden_layers, in_bn=in_bn, hid_bn=hid_bn, out_bn=out_bn)

    def _make_ff(self, dropout, in_size, hidden_size, hidden_layers, in_bn=True, hid_bn=True, out_bn=True, out=True):

        def get_linear(in_size, hidden_size):
            l = torch.nn.Linear(in_size, hidden_size)
            torch.nn.init.xavier_normal_(l.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            return l

        def get_block(in_size, hidden_size, bn, act=True, drop=True):
            result = [
                torch.nn.BatchNorm1d(in_size) if bn else None,
                torch.nn.Dropout(p=dropout) if drop else None,
                get_linear(in_size, hidden_size),
                torch.nn.ReLU() if act else None,
            ]
            return result

        ff_seq = list()
        ff_seq.extend(get_block(in_size, hidden_size, bn=in_bn))
        for _ in range(hidden_layers): ff_seq.extend(get_block(hidden_size, hidden_size, bn=hid_bn))
        if out: ff_seq.extend(get_block(hidden_size, 1, bn=out_bn, act=False, drop=False))

        return Sequential(
            *ff_seq
        )

    def cuda(self, *args, **kwargs):
        ret = super().cuda(*args, **kwargs)
        self.is_cuda = True
        return ret

    def cpu(self, *args, **kwargs):
        ret = super().cpu(*args, **kwargs)
        self.is_cuda = False
        return ret

    def forward(self,
                finance_features,
                presentation_toks_np,
                question_1_toks_np,
                answer_1_toks_np,
                ):
        return self.ff(finance_features)


class TextModel_AverageTokenEmbeddings(FinanceModel):

    def __init__(self,
                 train_dataset:Union[CallsDataset_V2],
                 hidden_size,
                 embedding_size=None,
                 hidden_layers=None,
                 dropout=None,
                 sparse=False,
                 init_std=False,
                 pretrained_embeddings_name=None,
                 adjust_embeddings=True,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 make_ff=True,
                 ):
        super().__init__(
            train_dataset=train_dataset,
            make_ff=False
        )

        if pretrained_embeddings_name is not None:
            pretrained_embeddings = numpy.load(os.path.join(train_dataset.dataset_dir, '..', 'embeddings', pretrained_embeddings_name))
            voc_size, embedding_size = pretrained_embeddings.shape
            self.tok_lookup = torch.nn.Embedding(voc_size + len(SPECIAL_TOKEN_IDS), embedding_size, padding_idx=PAD, sparse=sparse)
            self.tok_lookup.weight.data[len(SPECIAL_TOKEN_IDS):] = torch.from_numpy(pretrained_embeddings)
            if not adjust_embeddings:
                self.tok_lookup.weight.requires_grad = False
        else:
            self.tok_lookup = torch.nn.Embedding(train_dataset.data_stats['max_id'], embedding_size, padding_idx=0, sparse=sparse)
        self.embedding_size = embedding_size
        if make_ff:
            self.ff = self._make_ff(dropout, train_dataset.data_stats['feature_size'] + 3*self.embedding_size, hidden_size, hidden_layers, in_bn=in_bn, hid_bn=hid_bn, out_bn=out_bn)

    def forward(self,
                finance_features,
                presentation_toks_np,
                question_1_toks_np,
                answer_1_toks_np,
                ):
        bs = finance_features.shape[0]
        presentation_toks_np = self.tok_lookup(presentation_toks_np.view(bs, -1)).mean(dim=1)
        question_1_toks_np = self.tok_lookup(question_1_toks_np.view(bs, -1)).mean(dim=1)
        answer_1_toks_np = self.tok_lookup(answer_1_toks_np.view(bs, -1)).mean(dim=1)

        # finance features are zeroed in the data loader if the config entry 'mask' in experiment_settings is configured, i.e. not empty
        # e.g.
        return self.ff(
            torch.cat([finance_features,
                       presentation_toks_np,
                       question_1_toks_np,
                       answer_1_toks_np,
                       ], dim=1)
        )


class Model_with_attention(TextModel_AverageTokenEmbeddings):

    def __init__(self,
                 train_dataset:Union[CallsDataset_V2],
                 log_text_and_att,
                 phrases_pickle_dict,
                 hidden_size=None,
                 embedding_size=None,
                 hidden_layers=None,
                 dropout=None,
                 sparse=False,
                 init_std=False,
                 pretrained_embeddings_name=None,
                 adjust_embeddings=True,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 make_ff=True,
                 ):
        super().__init__(
                 train_dataset,
                 hidden_size,
                 embedding_size,
                 hidden_layers,
                 dropout,
                 sparse,
                 init_std,
                 pretrained_embeddings_name,
                 adjust_embeddings,
                 in_bn,
                 hid_bn,
                 out_bn,
                 make_ff,
                 )
        self.log_text_and_att = None
        if log_text_and_att is not None:
            self.log_text_and_att = open(log_text_and_att, 'w')
        self.phrases_pickle_dict = dict()

    def log_atts(self, batch_tokens_in, batch_atts, bs, predictions, max_nr=1):
        predictions = predictions.data.view(-1).cpu()
        calls_parts = [
            'PRESENTATION',
            'QUESTION 1',
            'ANSWER 1',
        ]

        for batch_nr in range(bs):
            for calls_part, (tok_ids, atts) in enumerate(zip(batch_tokens_in, batch_atts)):
                # tok_ids, atts = tok_ids_atts
                # atts_norm = atts[i].view(-1)
                atts_min = atts[batch_nr].min()
                atts_norm_scores = atts[batch_nr] - atts_min
                atts_norm_scores = (atts_norm_scores / (atts_norm_scores.max() - atts_min)).view(-1)

                # atts_norm_threshold = atts_norm_scores.data.mean()
                atts_norm_scores_list_sorted = list(sorted(atts_norm_scores.data.view(-1).cpu().numpy().tolist()))[::-1]
                atts_norm_threshold = atts_norm_scores_list_sorted[len(atts_norm_scores_list_sorted)//2]

                phrase = list()
                for tok, (atts_norm_score,tok_id) in enumerate(zip(atts_norm_scores.data.cpu(), tok_ids[batch_nr].data.cpu())):
                    if atts_norm_score > atts_norm_threshold:
                        if tok_id-8 > 0:
                            phrase.append(self.embedding_vocab.__getitem__(tok_id-8))
                        else:
                            phrase.append('')
                    else:
                        if len(phrase) > 0:
                            phrase_str = ' '.join(phrase)
                            if phrase_str not in self.phrases_pickle_dict:
                                self.phrases_pickle_dict[phrase_str] = list()
                            self.phrases_pickle_dict[phrase_str].append(predictions[batch_nr])
                            phrase = list()

                if batch_nr < max_nr:
                    self.log_text_and_att.write('<p>{}</p><p>\n'.format(calls_parts[calls_part % len(calls_parts)]))
                    for word, att in zip(
                            list(map(lambda x: self.embedding_vocab.__getitem__(x) if x >= 0 else '', map(lambda x: x - 8, tok_ids.data[batch_nr].cpu().numpy().tolist()))),
                            list(atts_norm_scores.data.cpu().numpy().tolist())
                    ):
                        self.log_text_and_att.write('<span style="background-color: rgb(255,{},{})">{}</span> '.format(int(255 * (1 - att)), int(255 * (1 - att)), word))
                    self.log_text_and_att.write('</p>\n')

    def _compute_repr_w_att(self, toks_in, att_filter, bn, bs):
        raise NotImplementedError()

    def compute_representations_w_att(self, tokens_in, bs):
        att_filters = [
            self.att_filter_pres,
            self.att_filter_ques,
            self.att_filter_ans,
        ]

        representations = list()
        attentions = list()

        for toks_in, att_filter, bn in zip(tokens_in, att_filters, self.bns):
            repr, mask, att = self._compute_repr_w_att(toks_in, att_filter, bn, bs)
            representations.append(repr)
            attentions.append(att)

        return representations, tokens_in, attentions, bs

    def forward(self,
                finance_features,  # finance features are zeroed in the data loader if mask is configured like this
                presentation_toks_np_in,
                question_1_toks_np_in,
                answer_1_toks_np_in,
                ):
        bs = finance_features.size()[0]
        tokens_in = [
            presentation_toks_np_in.view(bs, -1),
            question_1_toks_np_in.view(bs, -1),
            answer_1_toks_np_in.view(bs, -1)
        ]
        representations, tokens_in, attentions, bs = self.compute_representations_w_att(tokens_in, bs)
        prediction = self.ff(
            torch.cat([finance_features] + representations, dim=1)
        )
        if self.log_text_and_att is not None:
            self.log_atts(tokens_in, attentions, bs, prediction)
        return prediction


class TextModel_BidrectionalEmbeddings_and_Attention(Model_with_attention):

    def __init__(self,
                 train_dataset,
                 hidden_size,
                 embedding_size=None,
                 hidden_layers=3,
                 dropout=0.2,
                 sparse=False,
                 init_std=False,
                 pretrained_embeddings_name=None,
                 adjust_embeddings=True,
                 lstm_num_layers=None,
                 lstm_hidden_size=None,
                 lstm_dropout=0,
                 log_text_and_att=None,
                 phrases_pickle_dict=None,
                 embedding_vocab=None,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 ):
        super().__init__(train_dataset,
                         pretrained_embeddings_name=pretrained_embeddings_name,
                         adjust_embeddings=adjust_embeddings,
                         sparse=sparse,
                         make_ff=False,
                         log_text_and_att=log_text_and_att,
                         phrases_pickle_dict=phrases_pickle_dict,
                         )
        self.feature_size = train_dataset.data_stats['feature_size']
        self.ff = self._make_ff(dropout,
                                train_dataset.data_stats['feature_size'] + 3*self.embedding_size,
                                hidden_size,
                                hidden_layers,
                                in_bn=in_bn, hid_bn=hid_bn, out_bn=out_bn)
        if lstm_num_layers is None:
            lstm_num_layers = 2
        if lstm_hidden_size is None:
            self.lstm_hidden_size = self.embedding_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.bilstm = torch.nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.att_filter_pres = torch.nn.Linear(self.lstm_hidden_size*2, 1, bias=False)
        self.att_filter_ques = torch.nn.Linear(self.lstm_hidden_size*2, 1, bias=False)
        self.att_filter_ans = torch.nn.Linear(self.lstm_hidden_size*2, 1, bias=False)

        self.bns = [None, None, None, None, None, ]

        self.log_text_and_att = None
        self.embedding_vocab = None
        if embedding_vocab is not None:
            with open(embedding_vocab, 'rb') as f:
                self.embedding_vocab = pickle.load(f)

        self.flatten_params_called = dict()

    def _compute_repr_w_att(self, toks_in, att_filter, bn, bs):
        my_device_id = torch.cuda.device_of(self.tok_lookup.weight.data).idx
        if my_device_id not in self.flatten_params_called:
            print(my_device_id, self.flatten_params_called)
            self.flatten_params_called[my_device_id] = True
        self.bilstm.flatten_parameters()
        toks_in = toks_in.view(bs, -1)
        mask = (toks_in.data == 0).view(bs, -1)
        toks = self.tok_lookup(toks_in.view(bs, -1))
        toks_ctxt, _ = self.bilstm(toks)
        score = att_filter(toks_ctxt).view(bs, -1)
        score[mask] = -1e12
        att = torch.nn.functional.softmax(score, dim=1).view(bs, -1, 1)
        toks = (toks_ctxt * att).sum(dim=1)
        return toks, mask, att


class TextModel_BidrectionalEmbeddings_and_Attention_LateFusion_with_Finance(TextModel_BidrectionalEmbeddings_and_Attention):

    def __init__(self,
                 train_dataset,
                 hidden_size,
                 embedding_size=None,
                 hidden_layers=3,
                 dropout=0.2,
                 sparse=False,
                 init_std=False,
                 pretrained_embeddings_name=None,
                 adjust_embeddings=True,
                 lstm_num_layers=None,
                 lstm_hidden_size=None,
                 log_text_and_att=None,
                 embedding_vocab=None,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 ):
        super().__init__(
            train_dataset,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            hidden_layers=hidden_layers,
            dropout=dropout,
            sparse=sparse,
            init_std=init_std,
            pretrained_embeddings_name=pretrained_embeddings_name,
            adjust_embeddings=adjust_embeddings,
            lstm_num_layers=lstm_num_layers,
            lstm_hidden_size=lstm_hidden_size,
            log_text_and_att=log_text_and_att,
            embedding_vocab=embedding_vocab,
        )

        in_size = train_dataset.data_stats['feature_size']

        self._make_ff_late_fusion(dropout, in_size, 3*lstm_hidden_size*2, hidden_size, hidden_layers, in_bn, hid_bn, out_bn)
        self.out = Sequential(
            torch.nn.BatchNorm1d(hidden_size) if out_bn else None,
            torch.nn.Linear(hidden_size, 1)
        )

    def _make_ff_late_fusion(self, dropout, in_size_features, in_size_text, hidden_size, hidden_layers, in_bn, hid_bn, out_bn):
        self.ff_features = self._make_ff(0.0, in_size_features, hidden_size, 3, True, False, True, out=False)
        self.ff = self._make_ff(dropout, in_size_text, hidden_size, hidden_layers, in_bn, hid_bn, out_bn, out=False)

    def forward(self,
                finance_features,
                presentation_toks_np_in,
                question_1_toks_np_in,
                answer_1_toks_np_in,
                ):
        bs = finance_features.size()[0]
        tokens_in = [
            presentation_toks_np_in.view(bs, -1),
            question_1_toks_np_in.view(bs, -1),
            answer_1_toks_np_in.view(bs, -1),
        ]

        representations, tokens_in, attentions, bs = self.compute_representations_w_att(tokens_in, bs)
        prediction = self.out(self.ff_features(finance_features) + self.ff(torch.cat(representations, dim=1)))
        if self.log_text_and_att is not None:
            self.log_atts(tokens_in, attentions, bs, prediction)
        return prediction


class ModelWrappedWithMSELoss(nn.Module):

    def __init__(self, model_class):
        super(ModelWrappedWithMSELoss, self).__init__()
        self.model_class = model_class
        self.criterion = torch.nn.MSELoss(reduction='none')

    def init_model(self, args):
        self.model = self.model_class(**args)

    def forward(self, inputs, target):
        output = self.model(*inputs)
        target = target.view(-1)
        output = output.view(target.size(0), -1)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        select_loss = self.criterion(output, target) # loss for printing and to select the best model
        select_loss = select_loss.mean().view(1, 1)
        backward_loss = select_loss # loss for SGD
        return backward_loss, select_loss, output # return output to compute other metrics


class ModelWrappedWithMSELossAndMulticlassBatchLoss(nn.Module):

    def __init__(self, model_class):
        super(ModelWrappedWithMSELossAndMulticlassBatchLoss, self).__init__()
        self.model_class = model_class
        self.criterion = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def init_model(self, args):
        self.model = self.model_class(**args)

    def forward(self, inputs, target):
        output = self.model(*inputs)
        target = target.view(-1)
        output = output.view(target.size(0), -1)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        # print(output.size())
        # print(target.size())
        ce_output = output.view(target.size(0)//2, -1)
        _, target_max_idx = target.view(-1,2).max(dim=1)
        # print(ce_output.size())
        # print(target_max_idx.size())
        # if not self.model.training:
        #     print(list(zip(output.data.cpu().numpy().tolist(),target.data.cpu().numpy().tolist())))
        # select_loss = self.ce(ce_output, target_max_idx.view(target.size(0)//2)) # loss for printing and to select the best model
        select_loss = self.criterion(output, target).view(1, 1) * self.ce(ce_output, target_max_idx.view(-1)) # loss for printing and to select the best model
        backward_loss = select_loss  # loss for SGD
        return backward_loss, select_loss, output # return output to compute other metrics


class ModelWrappedWithBinaryCELoss(nn.Module):

    def __init__(self, model_class):
        super(ModelWrappedWithBinaryCELoss, self).__init__()
        self.model_class = model_class
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_model(self, args):
        self.model = self.model_class(**args)

    def forward(self, inputs, target):
        output = self.model(*inputs)
        target = target.view(-1)
        output = output.view(target.size(0), -1)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        ce_output = output.view(target.size(0)//2, -1)
        _, target_max_idx = target.view(-1,2).max(dim=1)
        select_loss = self.criterion(ce_output, target_max_idx.view(target.size(0)//2)) # loss for printing and to select the best model
        backward_loss = select_loss # loss for SGD
        return backward_loss, select_loss, output # return output to compute other metrics


class RankingHingeLoss(_Loss):

    def __init__(self, margin, size_average=True,):
        super().__init__(size_average=size_average)
        self.margin = margin

    def __call__(self, *args, **kwargs):
        batch, label = args
        label_mult = batch.data.new(batch.size())
        label_mult.fill_(1.)
        label_mult[range(label.size(0)),label.data] = -1.
        result = torch.nn.functional.relu((batch * label_mult).sum(1)+self.margin)
        if self.size_average:
            result = result.mean()
        return result

class MeanAbsoluteLoss(_Loss):

    def __init__(self, margin, size_average=True,):
        super().__init__(size_average=size_average)
        self.margin = margin

    def __call__(self, *args, **kwargs):
        batch, label = args
        label_mult = batch.data.new(batch.size())
        label_mult.fill_(1.)
        label_mult[range(label.size(0)),label.data] = -1.
        result = torch.nn.functional.relu((batch * label_mult).sum(1)+self.margin)
        if self.size_average:
            result = result.mean()
        return result


class ModelWrappedWithRankingLoss(nn.Module):

    def __init__(self, model_class):
        super(ModelWrappedWithRankingLoss, self).__init__()
        self.model_class = model_class
        self.criterion = RankingHingeLoss(0.1)

    def init_model(self, args):
        self.model = self.model_class(**args)

    def forward(self, inputs, target):
        output = self.model(*inputs)
        target = target.view(-1)
        output = output.view(target.size(0), -1)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        ce_output = output.view(-1, 2)
        _, target_max_idx = target.view(-1,2).max(dim=1)
        select_loss = self.criterion(ce_output, target_max_idx.view(-1)) # loss for printing and to select the best model
        backward_loss = select_loss # loss for SGD
        return backward_loss, select_loss, output # return output to compute other metrics


class ModelWrappedWithMSELossAndRankingLoss(nn.Module):

    def __init__(self, model_class):
        super(ModelWrappedWithMSELossAndRankingLoss, self).__init__()
        self.model_class = model_class
        self.criterion = torch.nn.MSELoss()
        self.ce = RankingHingeLoss(0.1)

    def init_model(self, args):
        self.model = self.model_class(**args)

    def forward(self, inputs, target):
        output = self.model(*inputs)
        target = target.view(-1)
        output = output.view(target.size(0), -1)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        ce_output = output.view(target.size(0)//2, -1)
        _, target_max_idx = target.view(-1,2).max(dim=1)
        select_loss = self.criterion(output, target).view(1, 1) * self.ce(ce_output, target_max_idx.view(-1)) # loss for printing and to select the best model
        backward_loss = select_loss  # loss for SGD
        return backward_loss, select_loss, output # return output to compute other metrics


class ModelWrappedWithMSELossAndKLDiv(nn.Module):

    def __init__(self, model_class):
        super(ModelWrappedWithMSELossAndKLDiv, self).__init__()
        self.model_class = model_class
        self.model = None
        self.criterion = torch.nn.MSELoss()
        self.att_reg = 0.001

    def init_model(self, args):
        self.model = self.model_class(**args)

    def forward(self, inputs, target):
        output, kl_divs = self.model(*inputs)
        target = target.view(-1)
        output = output.view(target.size(0), -1)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        select_loss = self.criterion(output, target).view(1, 1)
        backward_loss = select_loss + self.att_reg*kl_divs
        return backward_loss, select_loss, output


class Loss:
    ModelWrappedWithMSELoss = ModelWrappedWithMSELoss
    ModelWrappedWithMSELossAndMulticlassBatchLoss = ModelWrappedWithMSELossAndMulticlassBatchLoss
    ModelWrappedWithMSELossAndRankingLoss = ModelWrappedWithMSELossAndRankingLoss
    ModelWrappedWithBinaryCELoss = ModelWrappedWithBinaryCELoss
    ModelWrappedWithMSELossAndKLDiv = ModelWrappedWithMSELossAndKLDiv
    ModelWrappedWithRankingLoss = ModelWrappedWithRankingLoss

class Models:

    FinanceModel = FinanceModel # only finance (was Model_001)
    TextModel_AverageTokenEmbeddings = TextModel_AverageTokenEmbeddings # finance + averaging token embeddings (was Model_002)
    TextModel_BidrectionalEmbeddings_and_Attention = TextModel_BidrectionalEmbeddings_and_Attention # MDL2 text w bilstm + att (was Model_004)
    TextModel_BidrectionalEmbeddings_and_Attention_LateFusion_with_Finance = TextModel_BidrectionalEmbeddings_and_Attention_LateFusion_with_Finance # MDL2 text w bilstm + att + finance / late fusion

