import os
import torch
import pickle
import random
import numpy as np
import setproctitle
import time
from eval import testing
import time#
from nas4lfm import SuperNet
#from nas4ml import SuperNet
#from nas4bx import SuperNet
from ColdNAS_options import config
os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_device']
setproctitle.setproctitle('rec_cold_start')


def training(model, train_dataset,test_dataset, batch_size, num_epoch):
    if config['use_cuda']:
        model.cuda()

    training_set_size = len(train_dataset)
    model.train()
    best=100
    bestn3=0
    bestn5=0
    bestmae=0
    ba=0
    for epoch in range(num_epoch):
        
        random.shuffle(train_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*train_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            model.global_update(supp_xs, supp_ys, query_xs, query_ys)
        #loss,P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10= testing(model, config, test_dataset)
        loss,mae,n3,n5= testing(model, config, test_dataset)
        if loss<best:
            #torch.save(model.state_dict(),'supernet.pt') 
            ba=model.alpha
            best=loss
            bestmae=mae
            bestn3=n3
            bestn5=n5
           # best_metric=[P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10]
        print('epoch:{}   loss:{} '.format(epoch,loss))
        #print('epoch:{}   loss:{}  mae:{}  ndcg3:{}   ndcg5:{}  '.format(epoch,loss,mae,n3,n5))

    return best,ba#bestmae,bestn3,bestn5#,best_metric

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    master_path= "./data/lastfm_20"
    trainsz= int(len(os.listdir("{}/training/log".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    supp_mps_s = []
    query_xs_s = []
    query_ys_s = []
    query_mps_s = []
    for idx in range(trainsz):
            supp_xs_s.append(pickle.load(open("{}/training/log/supp_x_{}.pkl".format(master_path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/training/log/supp_y_{}.pkl".format(master_path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/training/log/query_x_{}.pkl".format(master_path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/training/log/query_y_{}.pkl".format(master_path, idx), "rb")))
            supp_mp_data, query_mp_data = {}, {}
            
    train_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    
    testsz= int(len(os.listdir("{}/testing/log".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(testsz):
            supp_xs_s.append(pickle.load(open("{}/testing/log/supp_x_{}.pkl".format(master_path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/testing/log/supp_y_{}.pkl".format(master_path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/testing/log/query_x_{}.pkl".format(master_path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/testing/log/query_y_{}.pkl".format(master_path, idx), "rb")))
    test_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
    aloss=[]

    '''B=[]
    BN3=[]
    BN5=[]
    BM=[]'''
    #t1=time.clock()
        
    print(config)
    seed_everything(config['seed'])
    model=SuperNet(config)
    b,a=training(model, train_dataset, test_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch_search'] )
    print(b,a)
    del model
    #t2=time.clock()
    #print("time",t2-t1)

   


