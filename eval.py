
import random
import argparse
from tqdm import tqdm
import torch
from utils.scorer import *
import numpy as np
def testing_simple(trainer, opt, test_dataset):
    test_dataset_len = len(test_dataset)
    minibatch_size = 1
    a, b, c, d = zip(*test_dataset)
    trainer.eval()
    for i in range(test_dataset_len):
        try:
            supp_xs = list(a[minibatch_size * i:minibatch_size * (i + 1)])
            supp_ys = list(b[minibatch_size * i:minibatch_size * (i + 1)])
            query_xs = list(c[minibatch_size * i:minibatch_size * (i + 1)])
            query_ys = list(d[minibatch_size * i:minibatch_size * (i + 1)])
        except IndexError:
            continue

def testing(trainer, opt, test_dataset):
    test_dataset_len = len(test_dataset)
    #batch_size = opt["batch_size"]
    minibatch_size = 1
    a, b, c, d = zip(*test_dataset)
    trainer.eval()
    all_loss = 0
    all_mae=0
    ndcg3=[]
    mae=[]
    pre5 = []
    ap5 = []
    ndcg5 = []
    pre7 = []
    ap7 = []
    ndcg7 = []
    pre10 = []
    ap10 = []
    ndcg10 = []
    for i in range(test_dataset_len):
        try:
            supp_xs = list(a[minibatch_size * i:minibatch_size * (i + 1)])
            supp_ys = list(b[minibatch_size * i:minibatch_size * (i + 1)])
            query_xs = list(c[minibatch_size * i:minibatch_size * (i + 1)])
            query_ys = list(d[minibatch_size * i:minibatch_size * (i + 1)])
        except IndexError:
            continue
        #mae.append(trainer.query_mae(supp_xs, supp_ys, query_xs, query_ys).cpu().detach().numpy())
        
        test_loss,mae, recommendation_list = trainer.query_rec(supp_xs, supp_ys, query_xs, query_ys)
        recommendation_list=recommendation_list.tolist()
        #print(recommendation_list,query_ys)
        n3=nDCG(recommendation_list,query_ys[0].tolist(),3)
        n5=nDCG(recommendation_list,query_ys[0].tolist(),5)
        ndcg3.append(n3)
        ndcg5.append(n5)
        all_loss += test_loss
        all_mae+=mae
        '''add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre5, ap5, ndcg5, 5)
        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre7, ap7, ndcg7, 7)
        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre10, ap10, ndcg10, 10)'''

    all_loss=all_loss/test_dataset_len
    all_mae=all_mae/test_dataset_len
    n3=np.mean(ndcg3)
    n5=np.mean(ndcg5)
    #mae=np.mean(mae)
    '''mpre5, mndcg5, map5 = cal_metric(pre5, ap5, ndcg5)
    mpre7, mndcg7, map7 = cal_metric(pre7, ap7, ndcg7)
    mpre10, mndcg10, map10 = cal_metric(pre10, ap10, ndcg10)'''

    #return all_loss,mpre5, mndcg5, map5, mpre7, mndcg7, map7, mpre10, mndcg10, map10
    
    return all_loss,all_mae,n3,n5
