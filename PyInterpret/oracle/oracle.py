from ..util import util

class DataOracle(object):
    def __init__(self, data = None, **kwargs):
        self.data = data
        self.is_loaded = data is not None        
        self.features = self.create_feature_ids()
        self.quantile_map = self.create_quantiles(**kwargs)
        
    

    def generate_ball_around_point(self, x, n = 100, fixed_scale = None):
        
        if not self.is_loaded or fixed_scale:
            return util.get_uniform_ball_around_point(x, n, fixed_scale = fixed_scale)
        
        else:
            return util.get_scaled_ball_around_point(x, n)   
    
    def create_feature_ids(self):
        if not self.is_loaded:
            return None
    
    def create_quantiles(self,**kwargs):
        if not self.is_loaded:
            return None
        
    def sample_data(self, around = None, model = None, by = 'random'):
        if by == 'importance':
            pass
        
        elif by == 'scaled':
            pass
        
        elif by == 'random':
            pass
        
        elif by == 'around':
            pass