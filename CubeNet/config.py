basic_info = {'batch_size' : 1, # batch size when do prediction
    'device' : 'cpu', # or 'cuda'
    }

arr_info = {
            'median_app' : 9.44, # median aperture conducted from training dataset
            'vir_evt_depth' : 3, # depth of virtual source, unit: km
            'vir_vp' : 5, # virtual P-wave velocity, unit: km/s
            'vir_vs' : 3, # virtual S-wave velocity, unit: km/s
            'model_vaild_fs' : (100,500), # range of sampling rate in training dataset
            }

train_info = {
    'training_batch_size' : 16, # batch size during training
    'epoch' : 20, # training epoch
    'learning_rate' : 3e-4,
    'path_cubes' : './demo/demo_data', # path to save training dataset
    'path_saveModels' : './', # path to save model parameters
    'path_saveLossLog' : './', # path to save training loss
    }