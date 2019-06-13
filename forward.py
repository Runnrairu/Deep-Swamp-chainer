# -＊- coding: utf-8 -＊-
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
xp = chainer.cuda.cupy
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
    def __call__(self,train):

        if not train:
            t,W=self.ODEnet(self.T,self.N)
        elif self.task_name =="StochasticDepth":
            t,W =self.StochasticDepth(self.T,self.N)
        elif self.task_name=="Fukasawa":
            t,W=self.Fukasawa(self.T,self.N)
        else:
            print("task_name is invalid!")
            raise NotFoundError
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
        return t,W
    def EularMaruyama(self,T,N):
        delta_t= float(T)/(N+1)
        t=[delta_t]*(N+1)
        sigma=np.sqrt(delta_t)
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
 


def swish(x,gpu_id):

    if gpu_id<0:
        beta=np.ones(x.shape[1:],dtype=np.float32)
    else:
        beta=xp.ones(x.shape[1:],dtype=np.float32)
    return F.swish(x,beta)

        
        
class ResidualBlock(chainer.Chain):
   def __init__(self, channel,gpu_id):
       w = chainer.initializers.HeNormal(1e-2)
       super(ResidualBlock, self).__init__()
       
       with self.init_scope():
           self.conv1 = L.Convolution2D(channel, channel, 3,1,1, False, w)
           self.conv2 = L.Convolution2D(channel, channel, 3, 1, 1, False, w) 
           self.gpu_id=gpu_id
   def __call__(self, x):
       h = self.conv1(x) 
       h = swish(h,self.gpu_id)
       h = self.conv2(h)
       return  h
    

        
        

class SD(chainer.ChainList):

   def __init__(self, T,N,p_T,channel,gpu_id ):
        super(SD, self).__init__()

        with self.init_scope():
            self.T=T
            self.N=N
            self.p_T=p_T
            for _ in range(N):
                self.add_link(ResidualBlock(channel,gpu_id))

   def __call__(self, x,t,W,train):
       step=0
       for f in self:
           if train and W[step]==0:
               x=x
           elif train:        
               x = x+t[step]*f(x)
           else:#test
               x=x+p(step*self.T/(self.N+1),self.T,self.p_T)*t[step]*f(x)
           step +=1
           
       return x
    
        
            
            
            
            
class param_gen(chainer.Chain):
    def __init__(self,channel,hypernet,gpu_id):
        super(param_gen, self).__init__()
        with self.init_scope():
            self.flowcon1_param=L.Convolution2D(channel,channel,3,pad=1,stride=1)    
            self.flowcon2_param=L.Convolution2D(channel,channel,3,pad=1,stride=1) 
            if hypernet:
                self.hy1=L.Linear(1,100)
                self.hy2=L.Linear(100,2*channel+2)
        self.hypernet=hypernet
        self.channel=channel
        self.gpu_id=gpu_id
    def __call__(self,t):
        if self.hypernet:
            h_1,h_2,c1,c2=self.hypernet_t(t)
            W1=self.flowcon1_param.W*h_1.reshape(self.channel,1,1,1)
            W2=self.flowcon2_param.W*h_2.reshape(self.channel,1,1,1)
            b1=self.flowcon1_param.b*c1
            b2=self.flowcon2_param.b*c2
        else:#ODENET
            W1=self.flowcon1_param.W
            W2=self.flowcon2_param.W
            b1=self.flowcon1_param.b
            b2=self.flowcon2_param.b
        return W1,b1,W2,b2
        
    def hypernet_t(self,t):
        if self.gpu_id<0:
            t=np.array([[t]])
            t=t.astype(np.float32)
        else:
            t=xp.array([[t]])
            t=t.astype(xp.float32)
        h=self.hy1(t)
        h=swish(h,self.gpu_id)
        h=self.hy2(h)
        h=F.transpose(h)

        return h[0:self.channel],h[self.channel:2*self.channel],h[-2],h[-1]

    

class DeepSwamp(chainer.Chain):

    def __init__(self, T,N,p_T,channel,hypernet,gpu_id):
        super(DeepSwamp, self).__init__()
        
        with self.init_scope():
            self.T=T
            self.N=N
            self.p_T=p_T
            self.param=param_gen(channel,hypernet,gpu_id)
            self.gpu_id=gpu_id

    def __call__(self, x,t,W,train):
        step=0
        t_now=0
        for delta_t in t: 
            p_t=p(t_now,self.T,self.p_T)
            W1,b1,W2,b2 = self.param(t_now)
            f_x=self.f(x,W1,b1,W2,b2)
            if train:        
                x = x+p_t*t[step]*f_x+np.sqrt(p_t*(1-p_t))*W[step]*f_x
            else:#test
                x=x+p_t*t[step]*f_x
            step +=1
            t_now +=delta_t
        return x
    def f(self,x,W1,b1,W2,b2):
        x=F.convolution_2d(x,W1,b1,1,1)
        x=swish(x,self.gpu_id)
        x=F.convolution_2d(x,W2,b2,1,1)
        return x
    
    
    

class flow_net(chainer.Chain):
    def __init__(self,task_name,hypernet,T,N,channel,train,gpu_id):
        super(flow_net,self).__init__()
        with self.init_scope():
            p_T=0.5
            if task_name=="ResNet":
                pass
            elif task_name=="StochasticDepth":
                self.flow=SD(T,N,p_T,channel,gpu_id)
            elif task_name=="ODEnet":
                pass    
            elif task_name=="SDEnet":
                pass
            elif task_name=="Milstein":
                pass
            elif task_name=="Fukasawa":
                self.flow=DeepSwamp(T,N,p_T,channel,hypernet,gpu_id)

            else:
                print("invalid!")
                raise NotFoundError
    def __call__(self,x,t,W,train):
        x=self.flow(x,t,W,train)
        return x
        
    




class model(chainer.Chain):
    def __init__(self, n_class,dense,channel,T,N,task_name,hypernet=False,first_conv=False,gpu_id=-1):
        super(model,self).__init__()
        
        with self.init_scope():
            self.channel=channel
            self.first_conv=first_conv
            self.timelist=time_list(T,N,task_name)
            w = chainer.initializers.HeNormal(1e-2)        
            if first_conv:
                self.firstconvf=L.Convolution2D(3,channel,3,1,1,False,w)
            
            self.timelist=time_list(T,N,task_name)
            self.dense=dense
            self.train=True
            self.flow=flow_net(task_name,hypernet,T,N,channel,self.train,gpu_id)
            self.gpu_id=gpu_id
            if self.dense:
                self.fc1=L.Linear(channel,dense)
                self.fc2=L.Linear(dense,n_class)
            else:
                self.fc=L.Linear(channel,n_class)
            
    def __call__(self,x):
        train=self.train
        if self.first_conv:
            x=self.firstconvf(x)
        else:
            x = F.pad(x,[(0,0),(0,self.channel-3),(0,0),(0,0)],"constant",constant_values=0)
        t,W=self.timelist(train)
        x=self.flow(x,t,W,train)
        x=F.average_pooling_2d(x, x.shape[2:])
        if self.dense:
            x=self.fc1(x)
            x=swish(x,self.gpu_id)
            y=self.fc2(x)
        else :
            y=self.fc(x)
        return y

