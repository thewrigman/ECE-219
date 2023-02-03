import numpy as np
import scipy.stats as stats

# optimizer based on the optimizer proposed in 
# "A consensus-based global optimization method for high dimensional machine learning problems"
# https://arxiv.org/pdf/1909.09249.pdf

class particle:

    def __init__(self, dimensions=3,bounds=[0,1],drift_rate=0.01,lr=0.01,noise_rate=0.01,seed=42):
        self.params = np.random.uniform(bounds[0],bounds[1],dimensions)
        self.drift_rate = drift_rate
        self.lr = lr
        self.noise_rate = noise_rate

    def update(self,x_bar):
        self.params = self.params - self.lr*self.drift_rate*(self.params-x_bar) + \
                self.noise_rate*np.sqrt(self.lr)*(self.params-x_bar)*np.random.normal(0,1,self.params.shape)
    
    def get_params(self):
        return self.params

class stochastic_optimizer:
    def __init__(self,particles=10,dimensions=3,bounds=[0,1],drift_rate=0.01,lr=0.01,noise_rate=0.01,beta=0.9,seed=42):
        self.particles = [particle(dimensions,bounds,drift_rate,lr,noise_rate,seed) for i in range(particles)]
        self.bounds = bounds
        self.lr = lr
        self.noise_rate = noise_rate
        self.drift_rate = drift_rate
        self.dimensions = dimensions
        self.beta = beta
        self.old_x_bar = None
        self.dimensions = dimensions


    def update(self,losses):
        # calculate x_bar 
        weights=np.exp(-self.beta*losses)
        x_bar=np.zeros(self.dimensions)
        for i in range(len(self.particles)):
            x_bar+=weights[i]*self.particles[i].get_params()
        x_bar/=np.sum(weights)
        # update particles
        for p in self.particles:
            p.update(x_bar)
        
    
    def get_params(self):
        return [p.get_params() for p in self.particles]
    
    def get_average_params(self):
        return np.mean(self.get_params(),axis=0)

if __name__=="__main__":
    #a test on a simple function
    def f(x):
        return x**2+100*np.sin(10*x)

    opt = stochastic_optimizer(particles=10,dimensions=1,bounds=[-100,100],drift_rate=0.1,lr=1,noise_rate=0.1,beta=0.9)
    losss=[]
    params=[]
    for i in range(1000):
        losses = np.array([f(p.get_params()) for p in opt.particles])
        if opt.update(losses):
            break
        params.append(opt.get_params())
        losss.append(np.mean(losses))
    import matplotlib.pyplot as plt
    plt.plot(losss)
    plt.savefig("losses.png")
    plt.close()
    params=np.array(params)
    for i in range(params.shape[1]):
        plt.plot(params[:,i])
    plt.savefig("params.png")
    
    