from ..util import util
import numpy as np

class DataOracle(object):
    def __init__(self, data = None, **kwargs):
        self.data = data
        self.is_loaded = data is not None        
        self.feature_ids = self.create_feature_ids()
        self.quantile_map = self.create_quantiles(**kwargs)
        
    
    def get_population(self):
        return self.data


    def get_useful_values(self, feature_id, model = None, method = 'quantiles'):
        if method == 'quantiles':
            return self.quantile_map[feature_id]

    def generate_ball_around_point(self, x, n = 100, fixed_scale = None):
        
        if not self.is_loaded or fixed_scale:
            return util.get_uniform_ball_around_point(x, n, fixed_scale = fixed_scale)
        
        else:
            return util.get_scaled_ball_around_point(x, n)   
    
    def create_feature_ids(self):
        if not self.is_loaded:
            return None
        else:
            return range(self.data.shape[1])
    
    def create_quantiles(self,**kwargs):
        if not self.is_loaded:
            return None
        else:
            Q = np.percentile(self.data, range(100), 0)
            return {i: Q[:, i] for i in self.feature_ids}
        
    def sample_data(self, around = None, model = None, by = 'random'):
        if by == 'importance':
            pass
        
        elif by == 'scaled':
            pass
        
        elif by == 'random':
            pass
        
        elif by == 'around':
            pass