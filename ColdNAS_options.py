config = {
    'seed':0,
    'cuda_device':'4',
    'hidden_units':[128,64,32],
    'embedding_dim': 32,
    
    
    'use_cuda': True,
    'use_res':1,
    'lr_i':1e-4,
    'lr_ii':1e-4,
   
    'batch_size': 32,
    'num_epoch_search': 100,
    'num_epoch_eval':100,

    #last.fm
    'ifeature_dim':3846,
    'ufeature_dim':1872,
    'rating_range_lfm':1,

    #movielens
    'field_size':7,
    'rating_range_ml':5,
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,

    #bookcrossing
    'field_size':7,
    'rating_range_bx':10,
    'num_author':102018,
    'num_pub':16806,
    'num_age_bx':10,
    'num_year':12,
    'num_loc':57035
}


