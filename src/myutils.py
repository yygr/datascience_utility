from os.path import getsize, exists
from os import makedirs
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.neighbors import KernelDensity
from scipy import signal, stats

def gaussian(x, a, b, c, d):
    return a*np.exp(-((x-b)/c)**2/2)+d

def log_gaussian(x, a, b, c, d):
    return a*np.exp(-((np.log(x)-b)/c)**2/2)/x+d

def log_normal(x, m, s):
    return np.exp(-((np.log(x)-m)/s)**2/2)/(np.sqrt(2*np.pi)*s*x)

def get_figratio(n):
    """
    1:1x1
    2:2x1
    3:3x1
    4:2x2
    5:3x2
    6:3x2
    7:4x2
    8:4x2
    9:3x3
    """
    d = np.array([n**0.5, n**0.5])
    if np.prod(np.floor(d)) < n:
        d[0] = np.ceil(d[0])
    if np.prod(np.floor(d)) < n:
        d[0] += 1
    return np.floor(d).astype('int')

bintypes = {
    'auto' : 'auto',
    'fd' : 'fd',
    'doane' : 'doane',
    'scott' : 'scott',
    'rice' : 'rice',
    'sturges' : 'sturges',
    'sqrt' : 'sqrt',
    'scalar50' : 50, 
    'scalar100' : 100, 
}

def plot_hist(data, xlog=False, ylog=False, estimate=False, density=False, kernel='gaussian', nb_max=int(10e6), p0=None):
    nr, nc = get_figratio(len(bintypes))
    if estimate=='kde':
        #kde = stats.gaussian_kde(data)
        kde = KernelDensity(kernel=kernel).fit(data[:, None])
    idx = np.arange(len(data))
    if len(data) > nb_max:
        idx = np.random.permutation(len(data))[:nb_max]
    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(16, 9))
    for _ax, _key in zip(ax.flat, bintypes.keys()):
        h, e = np.histogram(data[idx], bins=bintypes[_key])
        _ax.set_title(f'{_key}[{len(h)}]')
        if density:
            h = h/h.sum(keepdims=True)
        _ax.bar(e[:-1], h, e[1:]-e[:-1], align='edge')
        if xlog:
            _ax.set_xscale('log')
        if ylog:
            _ax.set_yscale('log')
        if estimate:
            x = (e[:-1]+e[1:])/2
            if estimate=='kde':
                y = np.exp(kde.score_samples(x[:, None]))
                label = 'kde'
            else:
                if p0 is None:
                    popt, pcov = curve_fit(estimate, x, h)
                else:
                    popt, pcov = curve_fit(estimate, x, h, p0=p0)
                y = estimate(x, *popt)
                label = f'{estimate.__name__}'
                for _p in popt:
                    label += f'\n{_p}'
                s = r2_score(h, y)
                label += f'\nR2:{s:.3f}'
            _ax.plot(x, y, 'r', label=label)
            _ax.legend()
    plt.tight_layout()
    return fig, ax

def cumulative_bins(data, bins=10, eps=None):
    data = data.reshape(-1)
    data = data[np.argsort(data)]
    idx = np.linspace(0, len(data)-1, bins+1, endpoint=True).astype('int')
    edge = np.unique(data[idx])
    if not eps is None:
        edge[0] -= eps
    return edge

def get_continuous(check):
    s = np.where(np.logical_and(~check[:-1], check[1:]))[0]
    e = np.where(np.logical_and(check[:-1], ~check[1:]))[0]
    s = np.r_[0, s] if check[0] else s
    e = np.r_[e, len(check)-1] if check[-1] else e 
    return s, e

def get_delay_im(na, window=100, stride=5, return_index=False, method='fft'):
    _w, _s = window, stride
    _idx = np.arange(0, len(na)-_w, _s)
    corr = []
    for i in _idx:
        _na = na[i:i+_w].copy()
        if _na.T[0].std()==0 or _na.T[1].std()==0:
            corr.append(np.zeros(len(_na)))
            continue
        _na = _na - _na.mean(0, keepdims=True)
        _std = _na.std(0, keepdims=True)
        _std[_std==0] = 1
        _na /= _std
        _corr = signal.correlate(_na.T[0], _na.T[1], mode='same', method=method)
        corr.append(_corr)
    corr = np.array(corr, dtype='float32')
    if return_index:
        return _idx+window, corr
    return corr

def gen_matrix(edges, data, value=None, func=[np.mean, np.std], return_labels=False, debug=False):
    labels = np.array([np.digitize(d, e, right=True) for e,d in zip(edges, data)]) -1
    _shape = [len(x)-1 for x in edges]
    _shape.append(len(edges))
    if debug:
        print(_shape)
    matrix = np.zeros(tuple(_shape))
    mask = np.ones(matrix.shape[:-1])
    if not value is None:
        stats = np.zeros(tuple(_shape[:-1]+[len(func)]))
    if debug:
        print(data.shape, matrix.shape, labels.shape, mask.shape)
        check = 0
    for i in zip(*np.where(mask)):
        _idx = np.arange(data.shape[-1])
        for j, k in enumerate(i):
            _tmp = np.where(labels[j][_idx]==k)[0]
            _idx = _idx[_tmp]
            if len(_idx)==0:
                break
        if len(_idx)==0:
            continue
        for j in range(len(i)):
            _data = data[j][_idx]
            if len(_data)>0:
                matrix[i][j] = _data.mean()
        if debug:
            print(i, len(_idx), matrix[i], end='\n')
            check += len(_idx)
        if value is None:
            continue
        for j, _f in enumerate(func):
            stats[i][j] = _f(value[_idx])
    if debug:
        print(check)
    if return_labels:
        if not value is None:
            return matrix, stats, labels
        return matrix, labels
    if not value is None:
        return matrix, stats
    return matrix