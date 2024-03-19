import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from re import I
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
from transformers import BertModel, ViTModel, BertTokenizer

def logging_func(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message)
    f.close()

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class EncoderImagePrecomp(nn.Module):
    def __init__(self):
        super(EncoderImagePrecomp, self).__init__()
        self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.embing = self.visual_encoder.config.hidden_size
        self.fc1 = nn.Linear(self.embing, 1024)
        self.fc2 = nn.Linear(self.embing, 1024)
        self.fc3 = nn.Linear(self.embing, 1024)
        self.fc4 = nn.Linear(self.embing, 1024)
        self.fc5 = nn.Linear(self.embing, 1024)
        self.fc6 = nn.Linear(self.embing, 1024)
        self.init_weights()

    def init_weights(self):
        for fc in [self.fc1, self.fc2, self.fc3,self.fc4, self.fc5, self.fc6]:
            r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
            fc.weight.data.uniform_(-r, r)
            fc.bias.data.fill_(0)

    def forward(self, images):
        img_emb = self.visual_encoder(pixel_values=images, output_hidden_states=True)
        img_output1 = img_emb.hidden_states[1]
        img_output2 = img_emb.hidden_states[3]
        img_output3 = img_emb.hidden_states[5]
        img_output4 = img_emb.hidden_states[7]
        img_output5 = img_emb.hidden_states[9]
        img_output6 = img_emb.hidden_states[11]
        
        image_emb1 = F.normalize(img_output1, dim=-1)
        features1 = self.fc1(image_emb1)
        image_emb2 = F.normalize(img_output2, dim=-1)
        features2 = self.fc2(image_emb2)
        image_emb3 = F.normalize(img_output3, dim=-1)
        features3 = self.fc3(image_emb3)
        
        image_emb4 = F.normalize(img_output4, dim=-1)
        features4 = self.fc4(image_emb4)
        image_emb5 = F.normalize(img_output5, dim=-1)
        features5 = self.fc5(image_emb5)
        image_emb6 = F.normalize(img_output6, dim=-1)
        features6 = self.fc6(image_emb6)
        return features1,features2,features3,features4,features5,features6
    
    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.embing = self.text_encoder.config.hidden_size
        self.fc1 = nn.Linear(self.embing, 1024)
        self.fc2 = nn.Linear(self.embing, 1024)
        self.fc3 = nn.Linear(self.embing, 1024)
        self.fc4 = nn.Linear(self.embing, 1024)
        self.fc5 = nn.Linear(self.embing, 1024)
        self.fc6 = nn.Linear(self.embing, 1024)
        self.init_weights()

    def init_weights(self):
        for fc in [self.fc1, self.fc2, self.fc3,self.fc4, self.fc5, self.fc6]:
            r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
            fc.weight.data.uniform_(-r, r)
            fc.bias.data.fill_(0)

    def forward(self,  targets, targets_attention):
        img_emb = self.text_encoder(input_ids=targets, attention_mask=targets_attention,output_hidden_states=True)
        img_output1 = img_emb.hidden_states[1]
        img_output2 = img_emb.hidden_states[3]
        img_output3 = img_emb.hidden_states[5]
        img_output4 = img_emb.hidden_states[7]
        img_output5 = img_emb.hidden_states[9]
        img_output6 = img_emb.hidden_states[11]
        
        image_emb1 = F.normalize(img_output1, dim=-1)
        features1 = self.fc1(image_emb1)
        image_emb2 = F.normalize(img_output2, dim=-1)
        features2 = self.fc2(image_emb2)
        image_emb3 = F.normalize(img_output3, dim=-1)
        features3 = self.fc3(image_emb3)
        
        image_emb4 = F.normalize(img_output4, dim=-1)
        features4 = self.fc4(image_emb4)
        image_emb5 = F.normalize(img_output5, dim=-1)
        features5 = self.fc5(image_emb5)
        image_emb6 = F.normalize(img_output6, dim=-1)
        features6 = self.fc6(image_emb6)
        return features1,features2,features3,features4,features5,features6
    
def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    # attn --> (batch, sourceL, queryL)
    # positive attention
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * lamda)
    attn_pos = l1norm(attn_pos, 1)
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)
    return attn_pos

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w2).clamp(min=eps)).squeeze()


def intra_relation(K, Q, xlambda):  # 自注意力
    """
    Q: (n_context, sourceL, d)
    K: (n_context, sourceL, d)
    return (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)
    attn = attn.view(batch_size * KL, KL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn


def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # print(attn.shape)
    # print(batch_size," ",queryL," ",sourceL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn


def xattn_score(images, captions, opt):
    """
    Note that this function is used to train the model with Discriminative Mismatch Mining.
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    max_pos = []
    max_neg = []
    max_pos_aggre = []
    max_neg_aggre = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    cap_len_i = torch.zeros(1, n_caption)
    n_region = images.size(1)
    batch_size = n_image
    N_POS_WORD = 0
    A = 0
    B = 0
    mean_pos = 0
    mean_neg = 0

    for i in range(n_caption):
        # Get the i-th text description
        n_word = 30
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_len_i[0, i] = n_word
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        # text-to-image direction
        t2i_sim = torch.zeros(batch_size * n_word).double().cuda()
        # --> (batch, d, sourceL)
        contextT = torch.transpose(images, 1, 2)

        # attention matrix between all text words and image regions
        attn = torch.bmm(cap_i_expand, contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * 0

        # # --------------------------------------------------------------------------------------------------------------------------
        # Neg-Pos Branch Matching
        # negative attention
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
        attn_neg = Row_max.lt(0).float()
        t2i_sim_neg = Row_max * attn_neg
        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)
        # positive attention
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.lambda_softmax)
        weiContext_pos = torch.bmm(attn_pos, images)
        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)
        # 2) positive effects based on relevance scores
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.lambda_softmax)
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r

        t2i_sim = t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)
        # # --------------------------------------------------------------------------------------------------------------------------

        # Discriminative Mismatch Mining
        # # --------------------------------------------------------------------------------------------------------------------------
        wrong_index = sim.sort(0, descending=True)[1][0].item()
        # Based on the correctness of the calculated similarity ranking, we devise to decide whether to update at each sampling time.
        if (wrong_index == i):
            # positive samples
            attn_max_row = torch.max(attn.reshape(batch_size * n_word, n_region).squeeze(), 1)[0].cuda()
            attn_max_row_pos = attn_max_row[(i * n_word): (i * n_word + n_word)].cuda()

            # negative samples
            neg_index = sim.sort(0)[1][0].item()
            attn_max_row_neg = attn_max_row[(neg_index * n_word): (neg_index * n_word + n_word)].cuda()

            max_pos.append(attn_max_row_pos)
            max_neg.append(attn_max_row_neg)
            N_POS_WORD = N_POS_WORD + n_word
            if N_POS_WORD > 200:  # 200 is the empirical value to make adequate samplings
                max_pos_aggre = torch.cat(max_pos, 0)
                max_neg_aggre = torch.cat(max_neg, 0)
                mean_pos = max_pos_aggre.mean().cuda()
                mean_neg = max_neg_aggre.mean().cuda()
                stnd_pos = max_pos_aggre.std()
                stnd_neg = max_neg_aggre.std()

                A = stnd_pos.pow(2) - stnd_neg.pow(2)
                B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
                C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(
                    2) * torch.log(stnd_neg / (opt.alpha * stnd_pos) + 1e-8)

                thres = opt.thres
                thres_safe = opt.thres_safe
                opt.stnd_pos = stnd_pos.item()
                opt.stnd_neg = stnd_neg.item()
                opt.mean_pos = mean_pos.item()
                opt.mean_neg = mean_neg.item()

                E = B.pow(2) - 4 * A * C
                if E > 0:
                    #     # A more simple way to calculate the learning boundary after alpha* adjustement
                    #     # In implementation, we can use a more feasible opt.thres_safe, i.e. directly calculate the empirical lower bound, as in the Supplementary Material.
                    #     # (note that alpha* theoretically unifies the opt.thres at training and opt.thres_safe at testing into the same concept)
                    opt.thres = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()
                    opt.thres_safe = (mean_pos - 3 * opt.stnd_pos).item()

                if opt.thres < 0:
                    opt.thres = 0
                if opt.thres > 1:
                    opt.thres = 0

                if opt.thres_safe < 0:
                    opt.thres_safe = 0
                if opt.thres_safe > 1:
                    opt.thres_safe = 0

                opt.thres = 0.7 * opt.thres + 0.3 * thres
                opt.thres_safe = 0.7 * opt.thres_safe + 0.3 * thres_safe
        # # --------------------------------------------------------------------------------------------------------------------------

        if N_POS_WORD < 200:
            opt.thres = 0
            opt.thres_safe = 0

        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities

class ContrastiveLoss(nn.Module):
    def __init__(self, opt, initial_margin=0.2, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = initial_margin  # 将 margin 定义为可学习的参数
        self.max_violation = max_violation
    def forward(self, scores, length,m):
        diagonal = scores.diag().view(length, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (m + scores - d1).clamp(min=0)
        cost_im = (m + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class NAAF(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_enc = EncoderImagePrecomp()
        self.txt_enc = BertEncoder()
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt)
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.a3 = nn.Parameter(torch.tensor(1.0))
        
        self.a4 = nn.Parameter(torch.tensor(1.0))
        self.a5 = nn.Parameter(torch.tensor(1.0))
        self.a6 = nn.Parameter(torch.tensor(1.0))
        
        self.b1 = nn.Parameter(torch.tensor(1.0))
        self.b2 = nn.Parameter(torch.tensor(1.0))
        self.b3 = nn.Parameter(torch.tensor(1.0))
        
        self.b4 = nn.Parameter(torch.tensor(1.0))
        self.b5 = nn.Parameter(torch.tensor(1.0))
        self.b6 = nn.Parameter(torch.tensor(1.0))
        
        params = [self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.b1,self.b2,self.b3,self.b4,self.b5,self.b6]
        params += list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        self.opt = opt
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()


    def train_emb(self, images, texts, ids, epoch, *args):
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        text_input = self.tokenizer(texts,
                       padding='max_length',  # 使用最大长度进行填充
                       truncation=True,       # 启用截断
                       max_length=30,         # 指定最大长度为30
                       return_tensors="pt")   # 返回 PyTorch 张量
        targets = text_input.input_ids
        targets_attention = text_input.attention_mask
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()
            targets_attention = targets_attention.cuda()
            ids = ids.cuda()
            
        img_emb1,img_emb2,img_emb3,img_emb4,img_emb5,img_emb6 = self.img_enc(images)
        cap_emb1,cap_emb2,cap_emb3,cap_emb4,cap_emb5,cap_emb6 = self.txt_enc(targets, targets_attention)
        
        img_emb = self.a1*img_emb1+self.a2*img_emb2+self.a3*img_emb3+self.a4*img_emb4+self.a5*img_emb5+self.a6*img_emb6
        self.optimizer.zero_grad()
        
        scores1 = xattn_score(img_emb1, cap_emb2, self.opt)
        scores2 = xattn_score(img_emb2, cap_emb2, self.opt)
        scores3 = xattn_score(img_emb3, cap_emb3, self.opt)
        scores4 = xattn_score(img_emb4, cap_emb4, self.opt)
        scores5 = xattn_score(img_emb5, cap_emb5, self.opt)
        scores6 = xattn_score(img_emb6, cap_emb6, self.opt)
        
        scores = self.b1*scores1+self.b2*scores2+self.b3*scores3+self.b4*scores4+self.b5*scores5+self.b6*scores6
        
        loss = self.criterion(scores,img_emb.size(0),0.2)
        
        self.logger.update('Le', loss.item())
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


def xattn_score_test(images, captions, opt):
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    batch_size = n_image
    opt.using_intra_info = True
    for i in range(n_caption):
        # Get the i-th text description
        n_word = 30
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        # --> (batch, d, sourceL)
        contextT = torch.transpose(images, 1, 2)
        # attention matrix between all text words and image regions
        attn = torch.bmm(cap_i_expand, contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * opt.thres_safe
        # attn_thres = attn - torch.ones_like(attn) * opt.thres
        # # --------------------------------------------------------------------------------------------------------------------------
        # Neg-Pos Branch Matching
        # negative attention
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
        if opt.using_intra_info:
            attn_intra = intra_relation(cap_i, cap_i, 5)
            attn_intra = attn_intra.repeat(batch_size, 1, 1)
            Row_max_intra = torch.bmm(attn_intra, Row_max.reshape(batch_size, n_word).unsqueeze(-1)).reshape(
                batch_size * n_word, 1)
            attn_neg = Row_max_intra.lt(0).double()
            t2i_sim_neg = Row_max * attn_neg
        else:
            attn_neg = Row_max.lt(0).float()
            t2i_sim_neg = Row_max * attn_neg
        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)
        # positive attention
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.lambda_softmax)
        weiContext_pos = torch.bmm(attn_pos, images)
        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)
        # 2) positive effects based on relevance scores
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.lambda_softmax)
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)
        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r
        t2i_sim = t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)
        # # --------------------------------------------------------------------------------------------------------------------------
        similarities.append(sim)
    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


