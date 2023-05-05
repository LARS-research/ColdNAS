
import torch
import torch.nn
import numpy as np
from copy import deepcopy
import random

from torch.autograd import Variable
from torch.nn import functional as F

def init_param(param_list):
    for param in param_list:
        torch.nn.init.normal(param,mean=0,std=0.01)

class AdaptiveNet(torch.nn.Module):
    def __init__(self, config,p_dim,relu=0):
        super(AdaptiveNet,self).__init__()
        self.ifeature_dim = config['ifeature_dim']
        self.ufeature_dim = config['ufeature_dim']
        self.embedding_dim = config['embedding_dim']
        self.rating_range=config['rating_range_lfm']
        self.context_dim=8*self.embedding_dim
        self.p_dim=p_dim
        self.relu=relu
        self.ctx_fc=torch.nn.Linear(self.embedding_dim*2+1,self.context_dim)
        self.out_layer=torch.nn.Linear(self.context_dim,self.p_dim)

    def forward(self,emb,ys):
        x=torch.cat((emb,ys.view(-1,1)),1)
        x=self.ctx_fc(x)
        x=F.leaky_relu(x)
        x=torch.mean(x,0)
        x=self.out_layer(x)
        if self.relu:x=F.relu(x)
        return x.view(4,-1)


class SuperNet(torch.nn.Module):
    def __init__(self, config):
        super(SuperNet,self).__init__()
        self.ifeature_dim = config['ifeature_dim']
        self.ufeature_dim = config['ufeature_dim']
        self.embedding_dim = config['embedding_dim']
        self.rating_range=config['rating_range_lfm']
        self.use_cuda=1
        self.hidden_units=torch.tensor(config['hidden_units'])
        ap_dim=int(self.hidden_units.sum())
        self.hidden_units=config['hidden_units']

        self.item_emb = torch.nn.Linear(in_features=self.ifeature_dim,out_features=self.embedding_dim)
        self.user_emb = torch.nn.Linear(in_features=self.ufeature_dim,out_features=self.embedding_dim)
        
        self.adp=AdaptiveNet(config,(self.embedding_dim*2+ap_dim)*4)
        

        self.fc1=torch.nn.Linear(self.embedding_dim*2,self.hidden_units[0])
        self.fc2=torch.nn.Linear(self.hidden_units[0],self.hidden_units[1])
        self.fc3=torch.nn.Linear(self.hidden_units[1],self.hidden_units[2])
        self.linear_out = torch.nn.Linear(self.hidden_units[2], 1)

        self.alpha=torch.nn.Parameter(torch.zeros(4,4).cuda())
        self.alpha.requires_grad=True

        init_param(list(self.parameters()))

        self.optim=torch.optim.Adam(self.parameters(),config['lr_i'])

    def modulate(self,x,maxp,minp,mutp,addp,alpha):
        x=alpha[0]*torch.maximum(x,maxp)+(1-alpha[0])*x
        x=alpha[1]*torch.minimum(x,minp)+(1-alpha[1])*x
        
        x=alpha[2]*x*mutp+(1-alpha[2])*x
        x=alpha[3]*(x+addp)+(1-alpha[3])*x
        
        return x

    def forward(self, xs,ys,xq, training = True):
        item_x = Variable(xs[:, 0:3846], requires_grad=False).float()
        user_x = Variable(xs[:, 3846:], requires_grad=False).float()
        item_emb = self.item_emb(item_x)
        user_emb = self.user_emb(user_x)
        emb = torch.cat((item_emb, user_emb), 1)

        p=self.adp(emb,ys)
        maxp=p[0]
        minp=p[1]
        mutp=F.relu(p[2])
        addp=p[3]

        alpha=torch.sigmoid(self.alpha)

        item_x = Variable(xq[:, 0:3846], requires_grad=False).float()
        user_x = Variable(xq[:, 3846:], requires_grad=False).float()
        item_emb = self.item_emb(item_x)
        user_emb = self.user_emb(user_x)
        emb = torch.cat((item_emb, user_emb), 1)
        d=0
        x=emb
        x=self.modulate(x,maxp[:self.embedding_dim*2],minp[:self.embedding_dim*2],mutp[:self.embedding_dim*2],addp[:self.embedding_dim*2],alpha[0])
        x=F.leaky_relu(x)
        x=self.fc1(x)
        d+=self.embedding_dim*2
        x=self.modulate(x,maxp[d:d+self.hidden_units[0]],minp[d:d+self.hidden_units[0]],mutp[d:d+self.hidden_units[0]],addp[d:d+self.hidden_units[0]],alpha[1]) 
        x=F.leaky_relu(x)
        x=self.fc2(x)
        d+=self.hidden_units[0]
        x=self.modulate(x,maxp[d:d+self.hidden_units[1]],minp[d:d+self.hidden_units[1]],mutp[d:d+self.hidden_units[1]],addp[d:d+self.hidden_units[1]],alpha[2])  
        x=F.leaky_relu(x)
        x=self.fc3(x)
        d+=self.hidden_units[1]
        x=self.modulate(x,maxp[d:d+self.hidden_units[2]],minp[d:d+self.hidden_units[2]],mutp[d:d+self.hidden_units[2]],addp[d:d+self.hidden_units[2]],alpha[2])               
        x=F.leaky_relu(x)
        x=self.linear_out(x)
        return self.rating_range*torch.sigmoid(x)

    def global_update(self, xs,ys,xq,yq):
        batch_sz = len(xs)
        loss=0
        self.optim.zero_grad()
        if self.use_cuda:
            for i in range(batch_sz):
                xs[i] = xs[i].cuda()
                ys[i] = ys[i].cuda()
                xq[i] = xq[i].cuda()
                yq[i] = yq[i].cuda()
        for i in range(batch_sz):
            y_pred=self.forward(xs[i],ys[i],xq[i],0).reshape(-1,1)
            #y_pred = torch.clip(y_pred,1e-6,1-1e-6)
            loss+=F.mse_loss(y_pred,yq[i].view(-1,1))
        loss=loss/batch_sz
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def query_rec(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = 1
        # used for calculating the rmse.
        losses_q = []
        losses_mae=[]
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            #query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], 0)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            loss_mae=F.l1_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
            losses_mae.append(loss_mae)
        losses_q = torch.stack(losses_q).mean(0)
        losses_mae = torch.stack(losses_mae).mean(0)
        output_list, recommendation_list = query_set_y_pred.view(-1).sort(descending=True)
        return losses_q.item(),losses_mae.item(), recommendation_list
