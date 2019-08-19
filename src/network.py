import renom as rm
import numpy as np
from os import listdir, makedirs, path, sep
from numpy import random
from func import VAE, Enc, Mahalanobis
from time import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from pdb import set_trace

class network():
    def __init__(
        self,
        input_shape, #(batch_size, input_size)
        latent_dim=2,
        epoch=5,
        units=1000,
        pre = None,
        dec = None,
        lr_ch = (5, 1.1),
        modeldir = 'model',
        outdir = 'result',
        cmap = plt.get_cmap('viridis'),
    ):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.epoch = epoch
        self.lr_ch = lr_ch
        self.shot = epoch//lr_ch[0]
        self.modeldir = modeldir
        self.outdir = outdir
        if not pre:
            pre = rm.Sequential([
                rm.Dense(units), rm.Relu()
            ])
            enc = Enc(pre, latent_dim)
        if not dec:
            dec = rm.Sequential([
                rm.Dense(units), rm.Relu(),
                rm.Dense(input_shape[-1]), rm.Sigmoid()
            ])
        self.ae = VAE(enc, dec, latent_dim)
        self.cmap = cmap

    def save(self, name):
        self.ae.save(name)

    def load(self, name):
        self.ae.load(name)

    def mini_batch(self, opt, x, inference=False, r=1.):
        self.ae.set_models(inference=inference)
        N = len(x)
        batch_data = np.zeros(self.input_shape, dtype='float32')
        z = np.zeros((N, self.latent_dim), dtype='float32')
        xz = np.zeros(x.shape, dtype='float32')
        batch_size = self.input_shape[0]
        code = ">" if not inference else "="
        if not inference:
            perm = random.permutation(N)
        else:
            perm = np.arange(N)
        batch_history = []
        for offset in range(0, N, batch_size):
            batch_start = time()
            _idx = perm[offset:offset+batch_size]
            _x = x[_idx]
            _len = len(_x)
            batch_data[:_len] = _x
            prepared_time = time()
            if not inference:
                with self.ae.train():
                    res = self.ae(batch_data)
            else:
                res = self.ae(batch_data)
            forwarded_time = time()
            l = self.ae.kl_loss + r*self.ae.reconE
            if not inference:
                l.grad().update(opt)
            end_time = time()
            if not inference:
                z[_idx] = self.ae.z[:_len].as_ndarray()
            else:
                z[_idx] = self.ae.enc.zm[:_len].as_ndarray()
            xz[_idx] = res[:_len].as_ndarray()
            batch_history.append([
                float(self.ae.kl_loss.as_ndarray()), #0
                float(self.ae.reconE.as_ndarray()), #1
                prepared_time - batch_start,#2
                forwarded_time - prepared_time,#3
                end_time - forwarded_time, #4
                end_time - batch_start, #5
            ])    
            mean = np.array(batch_history).mean(0)
            print_str = '{}{:5d}/{:5d}'.format(code, offset, N)
            print_str += ' KL:{:.3f} ReconE:{:.3f}'.format(
                mean[0], mean[1] 
            )
            print_str += ' ETA:{:.1f}sec'.format((N-offset)/batch_size*mean[-1])
            print_str += ' {:.2f},{:.2f},{:.2f} {:>10}'.format(
                mean[3], mean[4], mean[5], '')
            print(print_str, flush=True, end='\r')
        return batch_history, z, xz

    def _plot_latent(self, dic, outname, suffix='', message=''):
        x, y, z, xz = dic['x'], dic['y'], dic['z'], dic['xz']
        y_list = np.unique(y)
        i_max = len(y_list)

        fig, ax = plt.subplots(figsize=(10,10))
        if self.latent_dim==1:
            bins = np.linspace(z.min(), z.max(), 100)
            _mean = np.power(x,xz,2).mean(1)
            hist, edge = np.histogram(z, bins=bins)
            for i, l in enumerate(y_list):
                idx = np.where(y==l)[0]
                ax.bar(
                    edge[:-1], hist[idx], 
                    align='edge', width=edge[1:]-edge[:-1],
                    color=self.cmap((i+1)/i_max)
                )
        else:
            if self.latent_dim>2:
                z = TSNE().fit_transform(z)
            c = np.zeros(len(y))
            for i, l in enumerate(y_list):
                idx = np.where(y==l)[0]
                c[idx] = (i+1)/i_max
            ax.scatter(
                z[:,0], z[:,1],
                alpha=0.5, c=self.cmap(c)
            )
        plt.title(message)
        plt.tight_layout()
        outdir = '/'.join(outname.split('/')[:-1])
        if not path.exists(outdir):
            makedirs(outdir)
        plt.savefig(outname)
        plt.close()

    def _epoch(self, opt, x_train, x_test, y_train, y_test):
        history = []
        for e in range(1, self.epoch+1):
            code = '#'
            if e%self.shot==0:
                opt._lr /= self.lr_ch[1]
                self.ae.sigma /= self.lr_ch[1]
                code = '*'
                
            batch_history, z_train, xz_train = self.mini_batch(
                opt, x_train
            )
            train_mean = np.array(batch_history).mean(0)
            test_history, z_test, xz_test = self.mini_batch(
                opt, x_test
            )
            test_mean = np.array(test_history).mean(0)
            print_str = '{}{:5d}/{:5d}'.format(
                code, e, self.epoch
            )
            print_str += ' KL:{:.3f} ReconE:{:.3f}'.format(
                train_mean[0], train_mean[1]
            )
            # learning snapshot
            print(print_str)
            history.append([train_mean, test_mean])
            _history = np.array(history)
            fig, ax = plt.subplots(nrows=2)
            cat1 = ['KL', 'ReconE']
            cat2 = ['train', 'test']
            for i, _c1 in enumerate(cat1):
                for j, _c2 in enumerate(cat2):
                    ax[i].plot(_history[:,j,i], 
                        label='{}[{}]'.format(_c1,_c2))
                if len(_history[...,i]) > 2:
                    _min = _history[...,i].min()
                    _max = _history[...,i].max()
                    if _min>0 and _max/_min > 5e1:
                        ax[i].set_yscale('log')
            for i in range(len(cat1)):
                ax[i].legend()
                ax[i].grid()
            outdir = '{}/vae/{}'.format(
                self.outdir, self.latent_dim)
            if not path.exists(outdir):
                makedirs(outdir)
            outname = '{}/learningcurve.png'.format(outdir)
            plt.savefig(outname)
            plt.close()
            # save details
            _e = (e-1)%self.shot
            if _e!=self.shot-1 and not e==self.epoch:
                continue
            _e = (e-1)//self.shot
            print('-'*len(print_str))
            if self.latent_dim>2:
                continue
            self._plot_latent(
                {'x':x_train, 'y':y_train, 'z':z_train, 'xz':xz_train},
                outname='{}/{}_train.png'.format(outdir, _e+1)
            )
                
        return history

    def train(self, opt, x_train, x_test, y_train, y_test):
        history = self._epoch(opt, x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    data = np.load('data/mnist.npy')
    y_train = data[0][0]
    x_train = data[0][1].astype('float32')/255.
    y_test = data[1][0]
    x_test = data[1][1].astype('float32')/255.
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    random.seed(10)
    latent_dim = 2
    epoch = 20
    batch_size = 256
    opt = rm.Adam()
    ae = network(
        (batch_size, 28*28), 
        epoch=epoch,
        latent_dim=latent_dim,
        lr_ch=(10, 1.2)
    )
    ae.train(opt, x_train, x_test, y_train, y_test)

    _, z_train, xz_train = ae.mini_batch(
            opt, x_train, inference=True
    )
    f = Mahalanobis(z_train, y_train)
    _, z_test, xz_test = ae.mini_batch(
            opt, x_test, inference=True
    )
    f.set_th(0.9998)
    pred = np.argmin(f.predict(z_test), 1)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))


"""
(py3) yamagishiyouheinoMacBook-Pro-2:vae_classifier yamagishi$ python src/network.py
#    1/   10 KL:3.049 ReconE:23.938 ETA:0.0sec 0.02,0.08,0.10
#    2/   10 KL:5.000 ReconE:16.164 ETA:0.0sec 0.02,0.08,0.10
#    3/   10 KL:5.569 ReconE:14.898 ETA:0.0sec 0.02,0.11,0.13
#    4/   10 KL:5.854 ReconE:14.320 ETA:0.0sec 0.02,0.09,0.11
#    5/   10 KL:6.049 ReconE:13.919 ETA:0.0sec 0.02,0.09,0.11
#    6/   10 KL:6.173 ReconE:13.668 ETA:0.0sec 0.02,0.09,0.11
#    7/   10 KL:6.282 ReconE:13.469 ETA:0.0sec 0.02,0.09,0.11
#    8/   10 KL:6.364 ReconE:13.308 ETA:0.0sec 0.02,0.11,0.13
#    9/   10 KL:6.426 ReconE:13.165 ETA:0.0sec 0.02,0.09,0.11
#   10/   10 KL:6.473 ReconE:13.020 ETA:0.0sec 0.02,0.11,0.13
Computing DistL:6.611 ReconE:12.857 ETA:0.0sec 0.01,0.00,0.01
 1.5735626220703125e-05sec
[[ 958    0    6    2    1    7    1    1    4    0]0.00,0.01
 [   0 1057   11   26    1    1    3    5   29    2]
 [  11    0  980   11    5    4    3    6   12    0]
 [   3    0   21  934    0   25    0    7   16    4]
 [   4    0   14    0  909    2    3    5    5   40]
 [   8    0    6   24    2  830    2    2   13    5]
 [  21    2    6    1    6   22  895    0    5    0]
 [   1    2   44    8    2    1    0  923    4   43]
 [   7    0   15   35    5   23    2    4  867   16]
 [   6    2    5   14   41    8    0   14   14  905]]
             precision    recall  f1-score   support

          0       0.94      0.98      0.96       980
          1       0.99      0.93      0.96      1135
          2       0.88      0.95      0.92      1032
          3       0.89      0.92      0.90      1010
          4       0.94      0.93      0.93       982
          5       0.90      0.93      0.91       892
          6       0.98      0.93      0.96       958
          7       0.95      0.90      0.93      1028
          8       0.89      0.89      0.89       974
          9       0.89      0.90      0.89      1009

avg / total       0.93      0.93      0.93     10000

(py3) yamagishiyouheinoMacBook-Pro-2:vae_classifier yamagishi$
"""
