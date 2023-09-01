import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_score import GraphScore
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import numpy
# GCLSDA


class GCLSDA(GraphScore):
    def __init__(self, conf, training_set, test_set,i):
        super(GCLSDA, self).__init__(conf, training_set, test_set,i)
        self.i = i
        args = OptionConf(self.config['GCLSDA'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.model = GCLSDA_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # BPR损失函数
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # 对比学习损失
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                # 总损失作和
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # print('(GCLSDA)user:',u)
        u = self.data.get_user_id(u)
        # print('user_id:',u)
        # 通过user_id知道对应的嵌入表示*item嵌入表示的转置
        # print('sno嵌入表示',self.user_emb[u])
        # print('疾病嵌入表示',self.item_emb)
        # 183!
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class GCLSDA_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(GCLSDA_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        # 使每层输出方差尽可能相等的服从正态分布的参数初始化
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        # 张量拼接
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            # 二维矩阵的乘法torch.mm()；sparse针对稀疏矩阵
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        # print('item_all:',len(item_all_embeddings),item_all_embeddings)
        # len(item_all_embeddings) == 183 !
        return user_all_embeddings, item_all_embeddings
