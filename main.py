# -*- coding: utf-8 -*-
import load_data #my.py

import forward #my.py
import loss_func #my.py
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer import dataset
from chainer.datasets import  TupleDataset
from chainer.training import extensions
from chainer.training import triggers

import numpy as np
import scipy.stats
gpu_id=-1

class data_augment(dataset.DatasetMixin):
    def __init__(self,train = True):
        X_train, Y_train, X_test, Y_test = load_data.load()
        X_train, Y_train, X_test, Y_test =X_train.astype(np.float32),Y_train.astype(np.int32),X_test.astype(np.float32),Y_test.astype(np.int32)
        X_train, X_test = scipy.stats.zscore(X_train),scipy.stats.zscore(X_test)
        if train:
            self.data =TupleDataset(X_train,Y_train)
        else:
            self.data =TupleDataset(X_test,Y_test)
        self.train=train
        
    def __len__(self):
        return len(self.data)        
    
    def get_example(self, i):
        x,t= self.data[i]
        
        if self.train:
            # random crop
            pad_x = np.zeros((3, 40, 40), dtype=np.float32)
            pad_x[:, 4:36, 4:36] = x
            top = np.random.randint(0, 8)
            left = np.random.randint(0, 8)
            x = pad_x[:, top:top+32, left:left+32]
            # horizontal flip
            if np.random.randint(0, 1):
                x = x[:, :, ::-1]
            #random mask
            if False:
                x = x.transpose(1, 2, 0)
                h, w, _ = x.shape
                min_cut=1
                max_cut = 6
                x_size = np.random.randint(min_cut,max_cut)
                y_size = np.random.randint(min_cut,max_cut)
                x_offset = np.random.randint( w - x_size )
                y_offset = np.random.randint( h - y_size )
                x[x_offset:x_offset+x_size,y_offset:y_offset+y_size] = np.random.randn()
                x.transpose(2, 0, 1)
        
        #x=cuda.to_gpu(x)
        return x,t
    
def train(model_object, batchsize=100, gpu_id=gpu_id, max_epoch=200):

    # 1. Dataset
    train, test = data_augment(), data_augment(False)

    # 2. Iterator
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    # 3. Model
    
    #model = L.Classifier(model_object,lossfun=loss_func.loss_(model_object))
    model = L.Classifier(model_object)

    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    # 4. Optimizer
    optimizer = optimizers.MomentumSGD(0.05)
    optimizer.setup(model)

    # 5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='model/{}_cifar10_result'.format(model_object.__class__.__name__))
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=triggers.ManualScheduleTrigger([30,60,90,120,150,180,210,240,270,300],'epoch'))
    trigger = triggers.MaxValueTrigger('validation/main/accuracy', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=trigger)
    trainer.extend(extensions.snapshot_object(
            model, 'model/model_iter_{.updater.iteration}'), trigger=trigger)
    trainer.extend(extensions.snapshot_object(
            optimizer, 'model/optimizer_iter_{.updater.iteration}'), trigger=trigger)
    
    # 7. Evaluator

    class TestModeEvaluator(extensions.Evaluator):

        def evaluate(self):
            model = self.get_target('main')
            ret = super(TestModeEvaluator, self).evaluate()
            return ret

    trainer.extend(extensions.LogReport())
    trainer.extend(TestModeEvaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.run()
    del trainer

    return model



task_name ="Fukasawa"

if __name__ == "__main__":
    
    T= 1
    N =52
    task_name = task_name
    hypernet= 0
    model = train(forward.FlowNet(10,T,N,task_name,hypernet),batchsize=128,gpu_id=gpu_id, max_epoch=300)
    
    
    
