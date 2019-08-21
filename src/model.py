import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import renom as rm
from time import time
from sklearn.metrics import r2_score

class Model(rm.Model):
    SERIALIZED = (
        '_scaling', 
        '_xmu', 
        '_xstd', 
        '_ymu', 
        '_ystd',
    )
    def __init__(
        self,
        # learning parameter
        seed = 42,
        epoch = 1000,
        batch = 1024,
        opt = [rm.Adam, {'lr':0.01}],
        anneal = [],
        
        # log parameter
        modelfile = None,
        logging_curve = False,
        
        #private variables
        arch = None,
       
        #other variables
        *args,
        **kwargs
    ):
        super().__init__()
        self.__name__ = 'Model'
        self.seed = seed
        self.epoch = epoch
        self.batch = batch
        self.opt = opt[0](**opt[1])
        self.logging_curve = logging_curve
        self.anneal = anneal
        self.arch = arch
        self._gen_model()
        self._scaling = None
        self._xmu = 0
        self._xstd = 1
        self._ymu = 0
        self._ystd = 1
    
    def save(self, fname):
        #self._scaling = json.dumps(self.scaling)
        #self._arch = json.dumps(self.arch)
        super().save(fname)
        
    def load(self, fname):
        super().load(fname)
        #self.arch = json.dumps(self._arch)

    def _gen_model(self):
        depth = self.arch['depth']
        unit = self.arch['unit']
        # excluding mini-batch size
        input_shape = self.arch['input_shape']
        output_shape =  self.arch['output_shape']
        seq = []
        for i in range(depth):
            seq.append(rm.Dense(unit))
            seq.append(rm.Relu())
            if i < 1 or i==depth-1:
                seq.append(rm.BatchNormalize())
        seq.append(rm.Dense(output_shape))
        self._model = rm.Sequential(seq)
        
    def _forward(self, x):
        #print(x.shape)
        return self._model(x)
    
    def get_shape(self, n, shape):
        if isinstance(shape, int):
            return (n, shape)
        return tuple([n]+list(shape))
    
    def _set_inference(self, inference):
        self._model.set_models(inference=inference)
    
    def _train(self, x, idx, y):
        with self._model.train():
            z = self._forward(x)
            return z, self.arch['loss'](z[:len(idx)], y)
            #return z, rm.mean_squared_error(z[:len(idx)], y)
            
        
    def forward(self, x, perm, y=None, opt=None):
        n = len(x)
        output_shape = self.arch['output_shape']
        batch_input_shape = self.get_shape(self.batch, self.arch['input_shape'])
        batch_output_shape = self.get_shape(self.batch, output_shape)
        pred_output_shape = self.get_shape(n, output_shape)
        if opt is None or y is None:
            self._set_inference(True)
            #self._model.set_models(inference=True)
        history = []
        pred = np.zeros(pred_output_shape)
        _batch = self.batch
        for i in np.arange(0, n, _batch):
            _idx = perm[i:i+_batch]
            _batch_x = np.zeros(batch_input_shape)
            _batch_x[:len(_idx)] = x[_idx]
            if not y is None:
                _batch_y =  y[_idx]
            if opt is None or y is None:
                z = self._forward(_batch_x)
                if not y is None:
                    loss = rm.mean_squared_error(z[:len(_idx)], _batch_y)
            else:
                z, loss = self._train(_batch_x, _idx, _batch_y)
                if 0:
                    with self._train():#_model.train():
                        z = self._forward(_batch_x)
                        loss = rm.mean_squared_error(z[:len(_idx)], _batch_y)
                grad = loss.grad()
                grad.update(opt)
            if not y is None:
                history.append(loss.as_ndarray())
            pred[_idx] = z[:len(_idx)].as_ndarray()
        #self._model.set_models(inference=False)
        self._set_inference(False)
        if y is None:
            return pred
        return np.array(history)
    

    def _get_target_axis(self, _shape):
        target_axis = [0]
        if isinstance(_shape, int):
            return tuple(target_axis)
        for i in range(2, len(_shape)+1):
            target_axis.append(i)
        return tuple(target_axis)
    
    def normalization(self, x, y):
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        #print("test", input_shape, output_shape)
        target_axis = self._get_target_axis(input_shape)
        #print(target_axis)
        _xmu = x.mean(axis=target_axis, keepdims=True)
        _xstd = x.std(axis=target_axis, keepdims=True)
        _xstd[_xstd==0] = 1
        target_axis = self._get_target_axis(output_shape)
        _ymu = y.mean(axis=target_axis, keepdims=True)
        _ystd = y.std(axis=target_axis, keepdims=True)
        _ystd[_ystd==0] = 1
        self._scaling = 'normalization'
        self._xmu = _xmu
        self._xstd = _xstd
        self._ymu = _ymu
        self._ystd = _ystd

    def meanstd_norm(self, x, y):
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        #print("test", input_shape, output_shape)
        target_axis = self._get_target_axis(input_shape)
        #print(target_axis)
        _xmu = x.mean(axis=target_axis, keepdims=True)
        _xstd = x.std(axis=target_axis, keepdims=True)
        _xstd[_xstd==0] = 1
        target_axis = self._get_target_axis(output_shape)
        _ymu = y.mean(axis=target_axis, keepdims=True)
        _ystd = y.std(axis=target_axis, keepdims=True)
        _ystd[_ystd==0] = 1
        self._scaling = 'normalization'
        self._xmu = _xmu
        self._xstd = _xstd
        self._ymu = _ymu
        self._ystd = _ystd
        
    def minmax_norm(self, x, y):
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        target_axis = self._get_target_axis(input_shape)
        _xmu = x.mean(axis=target_axis, keepdims=True)
        _xstd = x.std(axis=target_axis, keepdims=True)
        _xstd[_xstd==0] = 1
        target_axis = self._get_target_axis(output_shape)
        _ymu = y.mean(axis=target_axis, keepdims=True)
        _ystd = y.std(axis=target_axis, keepdims=True)
        _ystd[_ystd==0] = 1
        self._scaling = 'standardization'
        self._xmu = _xmu
        self._xstd = _xstd
        self._ymu = _ymu
        self._ystd = _ystd

    def standardization(self, x, y):
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        target_axis = self._get_target_axis(input_shape)
        _xmu = x.mean(axis=target_axis, keepdims=True)
        _xstd = x.std(axis=target_axis, keepdims=True)
        _xstd[_xstd==0] = 1
        target_axis = self._get_target_axis(output_shape)
        _ymu = y.mean(axis=target_axis, keepdims=True)
        _ystd = y.std(axis=target_axis, keepdims=True)
        _ystd[_ystd==0] = 1
        self._scaling = 'standardization'
        self._xmu = _xmu
        self._xstd = _xstd
        self._ymu = _ymu
        self._ystd = _ystd
    
    def _scaling_xy(self, x, y):
        if not x is None:
            x = (x-self._xmu)/self._xstd
        if not y is None:
            y = (y-self._ymu)/self._ystd
        return x, y
    def _descaling_y(self, y):
        return y*self._ystd + self._ymu
        
 
    def fit(self, x_train, y_train, x_test=None, y_test=None, fname=None, additional=False):
        if not additional:
            self.e = 1
            np.random.seed = self.seed
        if not self._scaling is None:
            x_train, y_train = self._scaling_xy(x_train, y_train)
            if not x_test is None:
                x_test, y_test = self._scaling_xy(x_test, y_test)

        history = []
        print_str = ''
        while self.e <= self.epoch:
            if self.e in self.anneal:
                self.opt._lr /= 2
            s = time()
            perm = np.random.permutation(len(x_train))
            train = self.forward(x_train, perm, y=y_train, opt=self.opt)
            if not x_test is None:
                perm = np.arange(len(x_test))
                test = self.forward(x_test, perm, y=y_test)
            _cp = time() - s
            print_str = ' '*len(print_str)
            print(print_str, end='\r')
            print_str = f'{self.e:05d}/{self.epoch:05d} {train.mean():.6f}'
            if not x_test is None:
                print_str += f' {test.mean():.6f}'
            print_str += f' @ {_cp:.2f}sec'
            print(print_str, end='\r', flush=True)
            if x_test is None:
                history.append([self.e, train.mean(), _cp])
            else:
                history.append([self.e, train.mean(), test.mean(), _cp])
            self.e += 1
        print('')
        if fname:
            history = np.array(history)
            plt.plot(history[:,0], history[:,1], label='train')
            if not x_test is None:
                plt.plot(history[:,0], history[:,2], label='test')
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Error')
            plt.grid()
            plt.savefig(fname)
            plt.clf()
            plt.close()

    def predict(self, x_test):
        if not self._scaling is None:
            x_test, _ = self._scaling_xy(x_test, None)
        pred = self.forward(x_test, np.arange(len(x_test)))
        if not self._scaling is None:
            pred = self._descaling_y(pred)
        return pred

class Fully(Model):
    def _gen_model(self):
        N = self.batch

        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        if 'debug' in self.arch.keys():
            debug = self.arch['debug']
        else:
            debug = False

        self.batch_input_shape = self.get_shape(N, input_shape)
        self.batch_output_shape = self.get_shape(N, output_shape)
        depth = self.arch['depth']
        unit = self.arch['unit']
       
        units = np.ones(depth+1)*unit
        _unit =  np.prod(output_shape)
        units[-1] = _unit
        units = units.astype('int')
        layer = [rm.Flatten()]
        for _unit in units:
            layer.append(rm.BatchNormalize())
            layer.append(rm.Relu())
            layer.append(rm.Dense(_unit))
        #layer = layer[:-1] + [rm.Dropout()] + [layer[-1]]
        self.fcnn = rm.Sequential(layer)
        
        if debug:
            x = np.zeros(self.batch_input_shape)
            for _layer in layer:
                x = _layer(x)
                print(x.shape, str(_layer.__class__).split('.')[-1])
            x = rm.reshape(x, self.batch_output_shape)
            print(x.shape)
  
    def _train(self, x, idx, y):
        with self.fcnn.train():
            x = self.fcnn(x)
        z = rm.reshape(x, self.batch_output_shape) 
        return z, rm.mean_squared_error(z[:len(idx)], y)
    
    def _set_inference(self, inference):
        self.fcnn.set_models(inference=inference)

    def _forward(self, x):
        x = self.fcnn(x)
        return rm.reshape(x, self.batch_output_shape)
    
    def fit(self, train, test=None, fname=None):
        self.lr_hist = []
        self.e = 1
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        offset_max = train.shape[2]-(input_shape[1]+output_shape[1])
        output_idx = self.arch['output_idx']
        
        def get_xy(b):
            offset = np.random.randint(0, offset_max, len(b))
            x, y = [], []
            for i, _offset in enumerate(offset):
                x.append(b[i][:, _offset:_offset+input_shape[1]])
                _offset += input_shape[1]
                _b = b[i][output_idx]
                y.append(_b[:, _offset:_offset+output_shape[1]])
            return np.array(x), np.array(y)
       
        if not test is None:
            x_test, y_test = get_xy(test)
            if not self._scaling is None:
                x_test, y_test = self._scaling_xy(x_test, y_test)

        def get_time(_sec):
            _day, _hour, _min = 0, 0, 0
            _1day = 3600*24
            if _sec > _1day:
                _day = _sec//_1day
                _sec -= _day*_1day
            if _sec > 3600:
                _hour = _sec//3600
                _sec -= _hour*3600
            if _sec > 60:
                _min = _sec//60
                _sec -= _min*60
            return int(_day), int(_hour), int(_min), int(_sec)

        history, _time = [], []
        print_str = ''
        while self.e <= self.epoch:
            if self.e in self.anneal:
                self.opt._lr /= 2
            self.lr_hist.append(self.opt._lr)
            s = time()
            x_train, y_train = get_xy(train)
            if not self._scaling is None:
                x_train, y_train = self._scaling_xy(x_train, y_train)
            perm = np.random.permutation(len(x_train))
           
            if 0:
                print(input_shape, output_shape)
                print(x_train.shape, y_train.shape)
            
            _train = self.forward(x_train, perm, y=y_train, opt=self.opt)
            if not test is None:
                perm = np.arange(len(x_test))
                _test = self.forward(x_test, perm, y=y_test)
            _cp = time() - s
            _time.append(_cp)
            print_str = ' '*len(print_str)
            print(print_str, end='\r')
            print_str = f'{self.e:05d}/{self.epoch:05d} {_train.mean():.6f}'
            if not test is None:
                print_str += f' {_test.mean():.6f}'
            print_str += f' @ {_cp:.2f}sec'
            _sec = np.array(_time).mean()*(self.epoch-self.e)
            _day, _hour, _min, _sec = get_time(_sec)
            if _day > 0:
                print_str += f' / {_day:d}d-'
            else:
                print_str += ' / '
            print_str += f'{_hour:02d}:{_min:02d}:{_sec:02d}'
            print(print_str, end='\r', flush=True)
            if test is None:
                history.append([self.e, _train.mean(), _cp])
            else:
                history.append([self.e, _train.mean(), _test.mean(), _cp])
            self.e += 1
        _sec = np.array(_time).sum()
        _day, _hour, _min, _sec = get_time(_sec)
        print(f'\nFinished at {_day:d}d-{_hour:02d}:{_min:02d}:{_sec:02d}')
        if fname:
            history = np.array(history)
            plt.plot(history[:,0], history[:,1], label='train', alpha=.6)
            if not x_test is None:
                plt.plot(history[:,0], history[:,2], label='test', alpha=.6)
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Error')
            plt.grid()
            plt.twinx()
            plt.plot(history[:,0], self.lr_hist, 'k', lw=1)
            plt.ylabel('learning rate')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()
            plt.close()

class CNN(Model):
    def _gen_model(self):
        N = self.batch

        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        if 'debug' in self.arch.keys():
            debug = self.arch['debug']
        else:
            debug = False

        self.batch_input_shape = self.get_shape(N, input_shape)
        self.batch_output_shape = self.get_shape(N, output_shape)
        depth = self.arch['depth']
        unit = self.arch['unit']
       
        units = np.ones(depth+1)*unit
        _unit =  np.prod(output_shape)
        units[-1] = _unit
        units = units.astype('int')
        layer = [rm.Flatten()]
        for _unit in units:
            layer.append(rm.BatchNormalize())
            layer.append(rm.Relu())
            layer.append(rm.Dense(_unit))
        #layer = layer[:-1] + [rm.Dropout()] + [layer[-1]]
        self.fcnn = rm.Sequential(layer)
        
        if debug:
            x = np.zeros(self.batch_input_shape)
            for _layer in layer:
                x = _layer(x)
                print(x.shape, str(_layer.__class__).split('.')[-1])
            x = rm.reshape(x, self.batch_output_shape)
            print(x.shape)
  
    def _train(self, x, idx, y):
        with self.fcnn.train():
            x = self.fcnn(x)
        z = rm.reshape(x, self.batch_output_shape) 
        return z, rm.mean_squared_error(z[:len(idx)], y)
    
    def _set_inference(self, inference):
        self.fcnn.set_models(inference=inference)

    def _forward(self, x):
        x = self.fcnn(x)
        return rm.reshape(x, self.batch_output_shape)
    
    def fit(self, train, test=None, fname=None):
        self.lr_hist = []
        self.e = 1
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        offset_max = train.shape[2]-(input_shape[1]+output_shape[1])
        output_idx = self.arch['output_idx']
        
        def get_xy(b):
            offset = np.random.randint(0, offset_max, len(b))
            x, y = [], []
            for i, _offset in enumerate(offset):
                x.append(b[i][:, _offset:_offset+input_shape[1]])
                _offset += input_shape[1]
                _b = b[i][output_idx]
                y.append(_b[:, _offset:_offset+output_shape[1]])
            return np.array(x), np.array(y)
       
        if not test is None:
            x_test, y_test = get_xy(test)
            if not self._scaling is None:
                x_test, y_test = self._scaling_xy(x_test, y_test)

        def get_time(_sec):
            _day, _hour, _min = 0, 0, 0
            _1day = 3600*24
            if _sec > _1day:
                _day = _sec//_1day
                _sec -= _day*_1day
            if _sec > 3600:
                _hour = _sec//3600
                _sec -= _hour*3600
            if _sec > 60:
                _min = _sec//60
                _sec -= _min*60
            return int(_day), int(_hour), int(_min), int(_sec)

        history, _time = [], []
        print_str = ''
        while self.e <= self.epoch:
            if self.e in self.anneal:
                self.opt._lr /= 2
            self.lr_hist.append(self.opt._lr)
            s = time()
            x_train, y_train = get_xy(train)
            if not self._scaling is None:
                x_train, y_train = self._scaling_xy(x_train, y_train)
            perm = np.random.permutation(len(x_train))
           
            if 0:
                print(input_shape, output_shape)
                print(x_train.shape, y_train.shape)
            
            _train = self.forward(x_train, perm, y=y_train, opt=self.opt)
            if not test is None:
                perm = np.arange(len(x_test))
                _test = self.forward(x_test, perm, y=y_test)
            _cp = time() - s
            _time.append(_cp)
            print_str = ' '*len(print_str)
            print(print_str, end='\r')
            print_str = f'{self.e:05d}/{self.epoch:05d} {_train.mean():.6f}'
            if not test is None:
                print_str += f' {_test.mean():.6f}'
            print_str += f' @ {_cp:.2f}sec'
            _sec = np.array(_time).mean()*(self.epoch-self.e)
            _day, _hour, _min, _sec = get_time(_sec)
            if _day > 0:
                print_str += f' / {_day:d}d-'
            else:
                print_str += ' / '
            print_str += f'{_hour:02d}:{_min:02d}:{_sec:02d}'
            print(print_str, end='\r', flush=True)
            if test is None:
                history.append([self.e, _train.mean(), _cp])
            else:
                history.append([self.e, _train.mean(), _test.mean(), _cp])
            self.e += 1
        _sec = np.array(_time).sum()
        _day, _hour, _min, _sec = get_time(_sec)
        print(f'\nFinished at {_day:d}d-{_hour:02d}:{_min:02d}:{_sec:02d}')
        if fname:
            history = np.array(history)
            plt.plot(history[:,0], history[:,1], label='train', alpha=.6)
            if not x_test is None:
                plt.plot(history[:,0], history[:,2], label='test', alpha=.6)
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Error')
            plt.grid()
            plt.twinx()
            plt.plot(history[:,0], self.lr_hist, 'k', lw=1)
            plt.ylabel('learning rate')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()
            plt.close()


class Classifier(Model):
    def _gen_model(self):
        N = self.batch

        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        if 'debug' in self.arch.keys():
            debug = self.arch['debug']
        else:
            debug = False

        self.batch_input_shape = self.get_shape(N, input_shape)
        self.batch_output_shape = self.get_shape(N, output_shape)
        depth = self.arch['depth']
        unit = self.arch['unit']
       
        units = np.ones(depth+1)*unit
        _unit =  np.prod(output_shape)
        units[-1] = _unit
        units = units.astype('int')
        layer = [rm.Flatten()]
        for _unit in units:
            layer.append(rm.BatchNormalize())
            layer.append(rm.Relu())
            layer.append(rm.Dense(_unit))
        #layer = layer[:-1] + [rm.Dropout()] + [layer[-1]]
        self.fcnn = rm.Sequential(layer)
        
        if debug:
            x = np.zeros(self.batch_input_shape)
            for _layer in layer:
                x = _layer(x)
                print(x.shape, str(_layer.__class__).split('.')[-1])
            x = rm.reshape(x, self.batch_output_shape)
            print(x.shape)
  
    def _train(self, x, idx, y):
        with self.fcnn.train():
            x = self.fcnn(x)
        z = rm.reshape(x, self.batch_output_shape) 
        return z, rm.softmax_cross_entropy(z[:len(idx)], y)
    
    def _set_inference(self, inference):
        self.fcnn.set_models(inference=inference)

    def _forward(self, x):
        x = self.fcnn(x)
        return rm.reshape(x, self.batch_output_shape)
    
    def fit(self, train, test=None, fname=None):
        self.lr_hist = []
        self.e = 1
        input_shape = self.arch['input_shape']
        output_shape = self.arch['output_shape']
        offset_max = train.shape[2]-(input_shape[1]+output_shape[1])
        output_idx = self.arch['output_idx']
        
        def get_xy(b):
            offset = np.random.randint(0, offset_max, len(b))
            x, y = [], []
            for i, _offset in enumerate(offset):
                x.append(b[i][:, _offset:_offset+input_shape[1]])
                _offset += input_shape[1]
                _b = b[i][output_idx]
                y.append(_b[:, _offset:_offset+output_shape[1]])
            return np.array(x), np.array(y)
       
        if not test is None:
            x_test, y_test = get_xy(test)
            if not self._scaling is None:
                x_test, y_test = self._scaling_xy(x_test, y_test)

        def get_time(_sec):
            _day, _hour, _min = 0, 0, 0
            _1day = 3600*24
            if _sec > _1day:
                _day = _sec//_1day
                _sec -= _day*_1day
            if _sec > 3600:
                _hour = _sec//3600
                _sec -= _hour*3600
            if _sec > 60:
                _min = _sec//60
                _sec -= _min*60
            return int(_day), int(_hour), int(_min), int(_sec)

        history, _time = [], []
        print_str = ''
        while self.e <= self.epoch:
            if self.e in self.anneal:
                self.opt._lr /= 2
            self.lr_hist.append(self.opt._lr)
            s = time()
            x_train, y_train = get_xy(train)
            if not self._scaling is None:
                x_train, y_train = self._scaling_xy(x_train, y_train)
            perm = np.random.permutation(len(x_train))
           
            if 0:
                print(input_shape, output_shape)
                print(x_train.shape, y_train.shape)
            
            _train = self.forward(x_train, perm, y=y_train, opt=self.opt)
            if not test is None:
                perm = np.arange(len(x_test))
                _test = self.forward(x_test, perm, y=y_test)
            _cp = time() - s
            _time.append(_cp)
            print_str = ' '*len(print_str)
            print(print_str, end='\r')
            print_str = f'{self.e:05d}/{self.epoch:05d} {_train.mean():.6f}'
            if not test is None:
                print_str += f' {_test.mean():.6f}'
            print_str += f' @ {_cp:.2f}sec'
            _sec = np.array(_time).mean()*(self.epoch-self.e)
            _day, _hour, _min, _sec = get_time(_sec)
            if _day > 0:
                print_str += f' / {_day:d}d-'
            else:
                print_str += ' / '
            print_str += f'{_hour:02d}:{_min:02d}:{_sec:02d}'
            print(print_str, end='\r', flush=True)
            if test is None:
                history.append([self.e, _train.mean(), _cp])
            else:
                history.append([self.e, _train.mean(), _test.mean(), _cp])
            self.e += 1
        _sec = np.array(_time).sum()
        _day, _hour, _min, _sec = get_time(_sec)
        print(f'\nFinished at {_day:d}d-{_hour:02d}:{_min:02d}:{_sec:02d}')
        if fname:
            history = np.array(history)
            plt.plot(history[:,0], history[:,1], label='train', alpha=.6)
            if not x_test is None:
                plt.plot(history[:,0], history[:,2], label='test', alpha=.6)
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Error')
            plt.grid()
            plt.twinx()
            plt.plot(history[:,0], self.lr_hist, 'k', lw=1)
            plt.ylabel('learning rate')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()
            plt.close()
