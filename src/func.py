from pdb import set_trace
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import chi2

import renom as rm

class Enc(rm.Model):
    def __init__(
        self, pre, latent_dim,
        output_act = None,
        ):
        self.pre = pre
        self.latent_dim = latent_dim
        self.zm_ = rm.Dense(latent_dim)
        self.zlv_ = rm.Dense(latent_dim)
        self.output_act = output_act
    def forward(self, x):
        hidden = self.pre(x)
        self.zm = self.zm_(hidden)
        self.zlv = self.zlv_(hidden)
        if self.output_act:
            self.zm = self.output_act(self.zm) 
            self.zlv = self.output_act(self.zlv)
        return self.zm

class VAE(rm.Model):
    def __init__(
            self, 
            enc,
            dec,
            latent_dim, 
            batch_size = None,
            sigma = 1.
        ):
        self.latent_dim = latent_dim
        self.enc = enc
        self.dec = dec
        self.batch_size = batch_size
        self.sigma = sigma
    
    def forward(self, x, eps=1e-3):
        nb = len(x)
        self.enc(x)
        e = np.random.randn(nb, self.latent_dim)*self.sigma
        self.z = self.enc.zm + rm.exp(self.enc.zlv/2)*e
        self.decd = self.dec(self.z)
        self.reconE = rm.mean_squared_error(self.decd, x)
        self.kl_loss = - 0.5 * rm.sum(
            1 + self.enc.zlv - self.enc.zm**2 -rm.exp(self.enc.zlv)
        )/nb
        self.vae_loss = self.kl_loss + self.reconE
        return self.decd

class Mahalanobis():
    def __init__(self, data, label):
        self.i_max = label.max() + 1
        self.labels = np.unique(label)
        self.d = data.shape[-1]
        #print('Computing the mean')
        #s = time.time()
        self.mu = np.array([
            data[np.where(label==x)[0]].mean(0) for x in self.labels])
        #print(' {}sec'.format(time.time() - s))
        #print('Computing Cov')
        #s = time.time()
        self.cov = np.array([
            np.cov(data[np.where(label==x)[0]].T) for x in self.labels])
        #print(' {}sec'.format(time.time() - s))
        #n()
        print('Computing Dist')
        s = time()
        self.comp_dist(data, label)
        print(' {}sec'.format(time() - s))
        #self.set_th()
    
    def stat(self):
        print('latent dimention = {}'.format(self.d))
        print('{} classifier'.format(self.i_max))

    def a(self, x, i):
        temp = x-self.mu[i]
        #return np.dot(np.dot(temp, np.linalg.inv(self.cov[i])), temp.T)
        return np.dot(temp, np.linalg.solve(self.cov[i], temp.T))
    
    def al(self, x):
        return [self.a(x, i) for i in range(self.i_max)]
    
    def comp_dist(self, data, label):
        dist = [] 
        if 0:
            for x in self.labels:
                sub = data[np.where(label==x)[0]]
                dist.append(np.array([self.al(x) for x in sub]))
                #dist.append(np.diagonal(np.dot(np.dot(sub,self.cov[i]),sub.T)))
            else:
                for x in self.labels:
                    sub = data[np.where(label==x)[0]]
                    sub_dist = []
                    for i, y in enumerate(self.labels):
                        temp = sub - self.mu[i]
                        sub_dist.append(np.diag(
                            np.dot(temp, 
                            np.linalg.sove(self.cov[i], temp.T))
                        ))
        self.dist = np.array(dist)
    
    def get_dist(self, data):
        res = np.zeros((len(data), self.i_max))
        for i in range(self.i_max):
            temp = data - self.mu[i]
            res[:,i] = np.diag(
                np.dot(temp,
                np.linalg.solve(self.cov[i], temp.T))
            )
        return res
        #return np.array([self.al(x) for x in data])

    def gamma(self,n):
        return np.prod(np.arange(1,n))

    def chi_squared(self, u, k, s):
        a = 2*s
        b = k//2
        t = u/a
        v = np.power(t,b-1)*np.exp(-t)/a/self.gamma(b)
        return v

    def comp_th(self, th):
        assert th <= 1, "{}:th must be lower than 1 or equal to 1".format(th)
        dth = 1 - th
        return chi2.isf(dth, self.d)

    def get_ths(self, ths):
        ths_ = np.sort(ths)
        acc = 0
        split = 1e6
        maxv = 100
        delta = maxv/split
        athl = []
        ath = 0
        pre = 0
        for dth in ths_:
            while acc < dth:
                check_value = '\r{}'.format(acc)
                sys.stdout.write(check_value)
                sys.stdout.flush()
                acc += self.chi_squared(ath, self.d, 1) * delta
                ath += delta
            athl.append(ath)
        print('')
        return np.array(athl)
    
    def set_th(self, th=0.001):
        th = self.comp_th(th)
        self.th = th 

    def predict(self, data, th=None):
        res = self.get_dist(data)
        if th is None:
            return res / self.th
        return res / th

    def predicts(self, data, ths):
        temp = self.get_dist(data)
        res = []
        for th in ths:
            res.append(temp/th)
        return np.array(res)

    def predict_prob(self, data):
        res = self.get_dist(data)
        prob_all = []
        for item in res:
            subprob = []
            for i, x in enumerate(item):
                distance = self.cumprob[i][0]
                prob = self.cumprob[i][1]
                if distance[-1] < x:
                    subprob.append(prob[-1])
                else:
                    subprob.append(prob[np.argmax(distance>x)-1])
            prob_all.append(np.array(subprob))
        return res/self.th, np.array(prob_all)

    def comp_cummlative_probability(self, bins=100):
        cumprob = []
        for i in range(self.dist.shape[0]):
            hist, x = np.histogram(np.sort(self.dist[i][:,i]), bins)
            cum_hist = np.array([hist[:j].sum() for j,_ in enumerate(hist)])
            cum_hist = 1 - cum_hist/cum_hist.max().astype('float')
            cumprob.append((x[:-1],cum_hist))
        self.cumprob = np.array(cumprob)
