import torch
import numpy as np
from functools import wraps

def map_elementwise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        container, idx = None, None
        for arg in args:
            if type(arg) in (list, tuple, dict):
                container, idx = type(arg), arg.keys() if type(arg) == dict else len(arg)
                break
        if container is None:
            for value in kwargs.values():
                if type(value) in (list, tuple, dict):
                    container, idx = type(value), value.keys() if type(value) == dict else len(value)
                    break
        if container is None:
            return func(*args, **kwargs)
        elif container in (list, tuple):
            get = lambda element, i: element[i] if type(element) is container else element
            return container(wrapper(*[get(arg, i) for arg in args], 
                                     **{key:get(value, i) for key, value in kwargs.items()}) 
                             for i in range(idx))
        elif container is dict:
            get = lambda element, key: element[key] if type(element) is dict else element
            return {key:wrapper(*[get(arg, key) for arg in args], 
                                **{key_:get(value_, key) for key_, value_ in kwargs.items()}) 
                    for key in idx}
    return wrapper


def get_batch_single(  batch_size, x, device):
    @map_elementwise
    def batch_mask(X, num):
        return np.random.choice(X.size(0), num, replace=False)
    @map_elementwise
    def batch(X, mask):
        return X[mask].requires_grad_(True).float().to(device)

    mask = batch_mask(x[2], batch_size)
   
    return batch(x, mask)

def get_batch(batch_size, train_xop, train_xon, train_xb, train_xif, device):
    @map_elementwise
    def batch_mask(X, num):
        return np.random.choice(X.size(0), num, replace=False)
    @map_elementwise
    def batch(X, mask):
        return X[mask].requires_grad_(True).float().to(device)

    mask1 = batch_mask(train_xop[-1], batch_size[0])
    mask2 = batch_mask(train_xon[-1],  batch_size[0])
    mask3 = batch_mask(train_xb[-1],  batch_size[1])
    mask4 = batch_mask(train_xif[-1], batch_size[2])


    return batch(train_xop, mask1),  batch(train_xon, mask2), batch(train_xb, mask3), batch(train_xif, mask4),


def get_full_batch(batch_size, left, right, interface, boundary_l, boundary_r, device):
 
    @map_elementwise
    def batch(X):
        return X.requires_grad_(True).float().to(device)

    return  batch(left), batch(right), batch(interface), batch(boundary_l), batch(boundary_r)


def batch_type(batch, get_batch=get_batch, get_full_batch=get_full_batch):
    if batch=='full':
        return get_full_batch
    else:
        return get_batch
    

def grad(y, x, create_graph=True, keepdim=False):
    '''
    y: [N, Ny] or [Ny]
    x: [N, Nx] or [Nx]
    Return dy/dx ([N, Ny, Nx] or [Ny, Nx]).
    '''
    N = y.size(0) if len(y.size()) == 2 else 1
    Ny = y.size(-1)
    Nx = x.size(-1)
    z = torch.ones_like(y[..., 0])
    dy = []
    for i in range(Ny):
        dy.append(torch.autograd.grad(y[..., i], x, grad_outputs=z, create_graph=create_graph)[0])
    shape = np.array([N, Ny])[2-len(y.size()):]
    shape = list(shape) if keepdim else list(shape[shape > 1])
    return torch.cat(dy, dim=-1).view(shape + [Nx])

@map_elementwise
def To_tensor(x,device):
    x=torch.tensor(x).float().requires_grad_(True).to(device)
    return x




