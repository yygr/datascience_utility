import numpy as np
def ratiospace_division(N, K, C=1., constraints=None):
    if K == 0:
        return np.ones(N)*(C/K)
    if constraints is None:
        cons = tuple([(0, C)]*N)
    else:
        cons = constraints
    measure = np.r_[np.ones(K)*(C/K),0]
    def _ratiospace_division(n, k):
        if n==1:
            return [measure[k:].sum()]
        res, _min, _max = [], cons[N-n][0], cons[N-n][1]
        for _k in range(K, k-1, -1):
            _c = measure[k:_k].sum()
            if _max < _c or measure[_k:].sum() < cons[N-n+1][0]:
                continue
            elif _c < _min:
                break
            comb = _ratiospace_division(n-1, _k)
            for _comb in comb:
                res.append(np.r_[_c, _comb])
        return np.array(res)
    res = _ratiospace_division(N, 0)
    if len(res)==0:
        return []
    return res
def nCk(idx, k=2):
    if isinstance(idx, int):
        idx = np.arange(idx)
    if k==1:
        return [[x] for x in idx]
    if len(idx) <= k:
        return idx
    _idx = idx.copy()
    _idx = list(_idx)
    i = _idx.pop()
    n_1Ck_1 = nCk(_idx, k=k-1)
    return [[i]+_c for _c in n_1Ck_1] + [nCk(_idx, k=k)]

def origin_vector(x, use_mean=False, return_m=False):
    n = x.shape[1]
    m = np.ones((n, n-1))
    for i, j in zip(*np.where(m)):
        if j%2==0:
            _c = np.cos(2*i*np.pi/n)
        else:
            _c = np.sin(2*i*np.pi/n)
        #_c = np.cos(2*i*np.pi/n-j*np.pi/2)
        #_c = np.cos((4*i-n*j)/2/n*np.pi)
        m[i, j] = _c
    if use_mean:
        origin = x.mean(0)
    else:
        origin = (np.ones(n)/n).reshape(1, -1)
    if return_m:
        return np.dot(x-origin, m), m
    return np.dot(x-origin, m)
