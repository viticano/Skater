import numpy as np

def get_uniform_ball_around_point(x, n=100, fixed_scale = 1.0):
    fixed_scale = fixed_scale or 1.0

    return np.random.normal(x, np.ones(*x.shape) * fixed_scale, size=(n, x.shape[0]))

def get_scaled_ball_around_point(x, n=100):
    pass 