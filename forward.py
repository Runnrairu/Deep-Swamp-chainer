# -＊- coding: utf-8 -＊-
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

def p(t,T,p_T):
    return 1-float(t)*(1-p_T)/T
    


class time_list(object):
    def __init__(self, T,N,task_name,hypernet):
        self.T=T
        self.N=N
        self.task_name=task_name
        self.hypernet=hypernet
    def __call__(self):
        if task_name=="ODENet" or task_name=="ResNet",task_name=="test":
            delta_t,delta_W=ODEnet(self.T,self.N,hypernet)
        elif task_name =="StochasticDepth":
            delta_t,delta_W =StochasticDepth(self.T,self.N,hypernet,p_T=0.5)
        elif task_name =="EularMaruyama" or task_name == "MilsteinNet":
            delta_t,delta_W=EularMaruyama(self.T,self.N,hypernet,p_T=0.5)
        elif task_name=="Fukasawa":
            delta_t,delta_W=Fukasawa(self.T,self.N,hypernet,p_T=0.5)
        elif task_name == "SDtest":
            delta_t,delta_W=SDtest(self.T,self.N,hypernet,p_T=0.5)
        else:
            print("task_name is invalid!")
        return delta_t,delta_W
    def ODEnet(self,T,N):
        delta_t=[float(T)/(N+1)]*(N+1)
        delta_W = [0]*(N+1)
        return delta_t,delta_W
    def StochasticDepth(self,T,N,p_T=0.5):
        del_t= float(T)/(N+1)
        delta_t=[del_t]*(N+1)
        delta_W = [0]*(N+1)
        t_now=0
        a=[0,1]
        for i in range(N+1):
            p_t= p(t_now)
            delta_W[i]= np.random.choice(a, size=None, replace=True, p=[p_t,1-p_t])
            t_now +=del_t 
        return delta_t,delta_W
    def 



class ReversibleBlock(chainer.Chain):

   
       w = chainer.initializers.HeNormal(1e-2)
       n_out=n_in
       super(ReversibleBlock, self).__init__()
       with self.init_scope():
           self.bn1_1 = L.BatchNormalization(n_mid)
           self.conv1 = L.Convolution2D(n_in, n_mid, 3,stride,1, False, w)
           self.bn1_2 = L.BatchNormalization(n_mid)
           self.bn1_3 = L.BatchNormalization(n_mid)
           self.K_1=self.conv1.W
           self.bn2_1 = L.BatchNormalization(n_mid)
           self.conv2 = L.Convolution2D(n_in, n_mid, 3, stride, 1, False, w)
           self.bn2_2 = L.BatchNormalization(n_mid)
           self.bn2_3 = L.BatchNormalization(n_mid)
           self.K_2=self.conv2.W

           

   def __call__(self, y,z):
       h = self.bn1_1(z)
       h = self.conv1(h)
       h = self.bn1_2(h) 
       h = F.relu(h)

       
       h = F.deconvolution_2d(h,self.K_1,stride=1,pad=1)
       h= self.bn1_3(h)
       y_next= y+h

       g = self.bn2_1(y_next)
       
       
       g = self.conv2(g)
       
       g = self.bn2_2(g)
       g = F.relu(g)
       
       g = F.deconvolution_2d(g,self.K_2,stride=1,pad=1)
       g= self.bn2_3(g)
       z_next = z-g
       return  y_next,z_next







class Block(chainer.ChainList):

   def __init__(self, n_in, n_mid, n_bottlenecks, stride=1):
       super(Block, self).__init__()
       
       for _ in range(n_bottlenecks):
           self.add_link(ReversibleBlock(n_in, n_mid))

   def __call__(self, y,z):
       for f in self:
           y,z = f(y,z)
       return y,z





class Hamiltonian(chainer.Chain):

   def __init__(self, n_class=10, n_blocks=[6,6,6]):
       super(Hamiltonian, self).__init__()

       with self.init_scope():
           w = chainer.initializers.HeNormal()
           
           self.con =L.Convolution2D(3, 32, 3, 1, 1, False, w)
           self.bn =  L.BatchNormalization(32)
           self.res1 = Block(16, 16, n_blocks[0], 1)
           self.res2 = Block(32, 32,  n_blocks[1], 1)
           self.res3 = Block(56, 56,  n_blocks[2], 1)
           #self.fc4 = L.Linear(112, 100)
           #self.fc5 = L.Linear(100, n_class)
           
           self.fc = L.Linear(112, n_class)
           

   def __call__(self, x):
       x_ = self.con(x)
       x_ = self.bn(x_)
       y = x_[:,0:16,:,:]
       z = x_[:,16:32,:,:]
       
       y,z = self.res1(y,z)
       y,z= F.average_pooling_2d(y,2,stride=2),F.average_pooling_2d(z, 2,stride=2)
       y,z = F.pad(y,[(0,0),(0,16),(0,0),(0,0)],"constant",constant_values=0),F.pad(z,[(0,0),(0,16),(0,0),(0,0)],"constant",constant_values=0)
       y,z = self.res2(y,z)
       y,z=  F.average_pooling_2d(y, 2,stride=2),F.average_pooling_2d(z, 2,stride=2)
       y,z= F.pad(y,[(0,0),(0,24),(0,0),(0,0)],"constant",constant_values=0),F.pad(z,[(0,0),(0,24),(0,0),(0,0)],"constant",constant_values=0)
       y,z = self.res3(y,z)
       h=F.concat((y,z),axis=1)
       h = F.average_pooling_2d(h, h.shape[2:])#global
       h = F.reshape(h,[-1, 112])
       #h = F.relu(self.fc4(h))
       #h = self.fc5(h)
       
       h = self.fc(h)
       return h

class Full_RevHamiltonian(chainer.Chain):
    def __init__(self, n_class=10, n_blocks=[6,6,6]):
        super(Full_RevHamiltonian, self).__init__()

        with self.init_scope():
            w = chainer.initializers.HeNormal()
           
            self.con =L.Convolution2D(3, 128, 3, 1, 1, False, w)
            self.res1 = Block(64, 64, n_blocks[0], 1)
            self.res2 = Block(64, 64, n_blocks[1], 1)
            self.res3 = Block(64, 64, n_blocks[2], 1)
            self.fc4 = L.Linear(32*32*128, 1000)
            self.fc5 = L.Linear(1000, n_class)
    def __call__(self, x):
        x_ = self.con(x)
        y = x_[:,0:64,:,:]
        z = x_[:,64:128,:,:]
       
        y,z = self.res1(y,z)
        y,z =self.res2(y,z)
        y,z=self.res3(y,z)
        h=F.concat((y,z),axis=1)
        
        h = F.reshape(h,[-1, 128*32*32])
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        return h
