import os
import torch


class Config(object):
    def __init__(self,dict_path, npz_data, category, maxlen, generate_maxlen,model_save_path,model_load_path=None):
        self.dict_path = dict_path
        self.npz_data = npz_data
        self.category = category
        self.lr = 0.0001
        self.weight_decay = 1e-5
        self.use_gpu = torch.cuda.is_available()
        self.epoch = 50
        self.batch_size = 256
        self.maxlen = maxlen
        self.model_save_path=model_save_path
        self.model_load_path=model_load_path
        self.generate_maxlen = generate_maxlen




opt_shi=Config(dict_path='./chinese-poetry/jsontang/',
           npz_data='./data/tang.npz',
           category='poet',
           maxlen=125,
           generate_maxlen=125,
           model_save_path='./modelsave/shi/', # 前缀
            model_load_path='./modelsave/shi/' #全称路径
           )

opt_ci = Config(dict_path='./chinese-poetry/jsonci/',
                npz_data='./data/ci.npz',
                category='ci',
                maxlen=140,
                generate_maxlen=140,
                model_save_path='./modelsave/ci/',
                model_load_path=None
            )