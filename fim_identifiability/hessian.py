import numpy as np

def hessian_func(func, g, l, h_g, h_l, **kwargs):
    f = func
    d_gg = (f(g+h_g, l, **kwargs) - 2*f(g, l, **kwargs) + f(g-h_g, l, **kwargs)) / (h_g**2)
    d_ll = (f(g, l+h_l, **kwargs) - 2*f(g, l, **kwargs) + f(g, l-h_l, **kwargs)) / (h_l**2)
    d_gl = (f(g+h_g, l+h_l, **kwargs) - f(g+h_g, l-h_l, **kwargs)
            - f(g-h_g, l+h_l, **kwargs) + f(g-h_g, l-h_l, **kwargs)) / (4*h_g*h_l)
    return np.array([[d_gg, d_gl], [d_gl, d_ll]])