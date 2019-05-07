# -＊- coding: utf-8 -＊-
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

def p(t,T,p_T):
    return 1-float(t)*(1-p_T)/T



def G(t):#Fukasawa_scheme
    return 1.0

def G_nm(n,t_now):
    return 1.0/(G(t_now)*n)


class time_list(object):
    def __init__(self, T,N,task_name):
        self.T=T
        self.N=N
        self.task_name=task_name
    def __call__(self):
        if task_name=="ODENet" or task_name=="ResNet",task_name=="test":
            t,W=ODEnet(self.T,self.N,hypernet)
        elif task_name =="StochasticDepth":
            t,W =StochasticDepth(self.T,self.N,p_T=0.5)
        elif task_name =="EularMaruyama" or task_name == "MilsteinNet":
            t,W=EularMaruyama(self.T,self.N)
        elif task_name=="Fukasawa":
            t,W=Fukasawa(self.T,self.N,p_T=0.5)
        elfi task_name == "SDtest":
            t,W=SDtest(self.T,self.N,p_T=0.5)
        
        else:
            print("task_name is invalid!")
        return t,W
    def ODEnet(self,T,N):
        t=[float(T)/(N+1)]*(N+1)
        W = [0]*(N+1)
        return t,W
    def StochasticDepth(self,T,N,p_T=0.5):
        del_t= float(T)/(N+1)
        t=[del_t]*(N+1)
        W = [0]*(N+1)
        t_now=0
        a=[0,1]
        for i in range(N+1):
            p_t= p(t_now,T,p_T)
            W[i]= np.random.choice(a, size=None, replace=True, p=[p_t,1-p_t])
            t_now +=del_t 
        return delta_t,delta_W
    def EularMaruyama(self,T,N):
        delta_t= float(T)/(N+1)
        t=[delta_t]*(N+1)
        sigma=np.sqrt(del_t)
        W = np.random.normal(0,sigma , N+1)
        return t,W
    def Fukasawa(self,T,N):
        n=N
        t = [0]*(N+1)
        W=[0]*(N+1)
        N_list=np.random.normal(0,1.0,[n+1])
        E_list=np.random.exponential(1.0,[n+1])
        t_now=0
        m=0
        d=1
        a= np.power(1+2.0/d,1+d/2.0)
        while(t_now < (T-a*G_nm(n,t_now)) and m<n):

            N = N_list[m]
            E = E_list[m]
            ab_N = np.absolute(N)
            Z = (ab_N*ab_N+2*E)/d
            G__nm = G_nm(n,t_now)
            delta_t = G__nm*a*np.exp(-Z)
            t[m]= delta_t
            t_now += delta_t
            #if t_now>T:

            #    t_now -= delta_t

            #    break
            W[m] = np.power(G__nm*a*d*Z*np.exp(-Z),0.5)*N/ab_N
            m+=1
        delta_euler_t = (T-t_now)/(n-m+1)
        sigma_euler_t = np.power(delta_euler_t,0.5)
        for i in range(n-m+1):
            t[m+i] = delta_euler_t
            W[m+i] = sigma_euler_t*N_list[m+i]
        return t,W
    def SDtest(self,T,N,p_T=0.5):
        t=[0]*(N+1)
        W = [0]*(N+1)
        t_now=0
        delta_t= float(T)/(N+1)
        for i in range(N+1):
            t[i]=p(t_now,T,p_T)
            t_now+=delta_t
        return t,W
        

class flowBlock(chainer.Chain):
    def __init__(self, channnel,depth,T,N,hypernet=0,task_name):
        super(flowBlock, self).__init__()
   
        

           

    def __call__(self,t,x):
       
        
        
       
       return  x+f(x)







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
