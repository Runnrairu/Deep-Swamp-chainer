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
        a=[1,0]#p_tの情報含めたほうが早そう
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
    def __init__(self, stride=1,pad=1):
        super(flowBlock, self).__init__()
        self.stride=stride
        self.pad=pad
    def __call__(self,x,delta_t,delta_W,t_now,W1,b1,W2,b2,SD=False,p_t=1,Mil=False):
        if not Mil and not SD:
            h=F.convolution_2d(x,W1,b1,self.stride,self.pad)
            h=F.swish(h,1.)
            h=F.convolution_2d(h,W2,b2,self.stride,self.pad)
                return x+delta_t*p_t*h+np.sqrt(p_t*(1-p_t))*h*delta_W

        elif SD:
            if delta_W = 0:
                return x
            else:
                h=F.convolution_2d(x,W1,b1,self.stride,self.pad)
                h=F.swish(h,1.)
                h=F.convolution_2d(h,W2,b2,self.stride,self.pad)
                return x+delta_t*h
        else:
            #Milstein

class FlowNet(chainer.Chain):
    def __init__(self, n_class,dense,channel,T,N,task_name,hypernet,first_conv=False,train_=True):
        super(FlowNet,self).__init__()
        self.channel=channel
        self.T=T
        self.N=N
        self.task_name=task_name
        self.hypernet=hypernet
        self.first_conv=first_conv
        self.timelist=time_list(T,N,task_name)
        w = chainer.initializers.HeNormal(1e-2)
        if first_conv:
            self.firstconvf=L.Convolution_2D(3,3*channel,3,1,1,False,w)
        self.param_gen=param_gen(3*channel,task_name,hypernet)# new class
        self.train=train_
        self.SD=False
        self.Mil=False
        if train_ and task_name=="StochasticDepth" :
            self.SD=True
        elif train_ and task_name == "Milstein":
            self.Mil=True
        if not train:
            if task_name=="StochasticDepth" or task_name=="EularMaruyama" or task_name=="Fukasawa" or task_name =="Milstein":
                task_name=="SDtest"
        self.timelist=time_list(T,N,task_name)
        self.dense=dense
        if dense:
            self.fc1=L.Linear(3*channel,dense)
            self.fc2=L.Linear(dense,n_class)
        else:
            self.fc=L.Linear(3*channel,class)
            
    def __call__(self,x,t=None):
        
        if self.first_conv:
            x=self.firstconvf(x)
        else:
            #タイル
        t,W=self.timelist()
        t_now=0
        for i in range(self.N+1):
            W1,b1,W2,b2=
            p_t=p(t_now,self.T,self.N)
            x=flowBlock(x,t[i],W[i],t_now,W1,b1,W2,b2,self.SD=False,p_t,self.Mil=False)
            t_now += t[i]
        x=F.average_pooling_2d(x, x.shape[2:])
        if self.dense:
            x=self.fc1(x)
            y=self.fc2(x)
        else self.dense:
            y=self.fc(x)
        if self.train:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t) 
        ekse:
            return y
        
            
            
            
