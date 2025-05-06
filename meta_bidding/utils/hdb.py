import numpy as np
import matplotlib.pyplot as plt

class step_solver():
    '''
    @inputs:
        x: [list] Price, shape: N (number of curve points)
        y: [list] Power, shape: N 
        xp: [list] shape: P (number of anchor points)
        yp: [list] shape: P
        upx: float, upper bound of x axis
        upy: float, upper bound of y axis
        lwx: float, lower bound of x axis
        lwy: float, lower bound of y axis
    @step:
        call the `step` function for doing one bottom up + top down update
    @outputs:
        call `visualize(dpi)` for plotting the result in matplotlib
        call `get_points` for getting price-quantity pairs
    '''
    def __init__(self,x,y,
                 xp = None,
                 yp = None,
                 lwx = -np.inf):
        self.x = np.array(x)
        self.y = np.array(y)
        self.upx = x[-1]
        self.upy = y[-1]
        self.pn = 10
        self.lwx = lwx
        self.lwy = -(y<0).any().astype('float')

        # if xp is None:
        #     xp = np.quantile(self.x,np.linspace(0.1,0.9,self.pn))
        # if yp is None:
        #     yp = np.quantile(self.y,np.linspace(0.7,1,self.pn))

        # Initialize the anchor points (Important!)
        if (y<0).any():
            power_anchor_list = np.array([-0.999,-0.95,-0.8,-0.4,0,0.2,0.4,0.6,0.8,0.999])
        else:
            power_anchor_list = np.array([0,0.2,0.4,0.6,0.8,0.85,0.9,0.95,0.98,0.999])

        # Find the closest anchor points
        id_anchor =  np.searchsorted(y,power_anchor_list,'left')
        id_anchor = np.clip(id_anchor,1,len(y)-1)
        xp = x[id_anchor] - 5
        yp = np.clip(y[id_anchor] + 0.05,None,1)



        self.xp = np.concatenate([[lwx],xp,[self.upx]]) 
        self.yp = np.concatenate([[self.lwy],yp,[self.upy]])
    
    def get_split_interval(self,i):
        return [[np.searchsorted(self.y,self.yp[i-1],'right'),np.searchsorted(self.x,self.xp[i],'left')],
                [np.searchsorted(self.x,self.xp[i],'right'),np.searchsorted(self.y,self.yp[i],'left')],
                [np.searchsorted(self.y,self.yp[i],'right'),np.searchsorted(self.x,self.xp[i+1],'left')]]

    def step(self):
        for rg in [range(1,self.pn-2),range(self.pn-2,1,-1)]:
            for i in rg:
                interval = self.get_split_interval(i)
                if interval[1][0]>=interval[2][1]:
                    self.yp[i] = 0.5*(self.yp[i-1]+self.yp[i+1])
                else:
                    self.yp[i] = np.average(self.y[interval[1][0]:interval[2][1]])
                left_middle_id = np.searchsorted(self.y,0.5*(self.yp[i]+self.yp[i-1]),'left')
                try:
                    self.xp[i] = self.x[left_middle_id]
                except:
                    self.xp[i] = self.x[-1]
                if self.xp[i]==self.xp[i-1]:
                    self.xp[i] = 0.5*(self.xp[i-1]+self.xp[i+1])
        self.yp = np.maximum.accumulate(self.yp)

    def visualize(self,dpi = None):
        plt.figure(dpi = dpi)
        plt.plot(self.y,self.x)
        for i in range(0,self.pn-1):
            plt.plot([self.yp[i],self.yp[i]],[self.xp[i],self.xp[i+1]], color='b',linewidth = 0.8)
        for i in range(0,self.pn-2):
            plt.plot([self.yp[i],self.yp[i+1]],[self.xp[i+1],self.xp[i+1]],'b--',linewidth = 0.8)

        plt.scatter(self.yp[:-1],self.xp[:-1],color = 'b',marker='.')
        plt.scatter(self.yp[:-1],self.xp[1:],color = 'w',marker='.',edgecolors='b')
        plt.ylabel("Price")
        plt.xlabel("Capacity")
        plt.title("fit mean squared error:"+str(np.round(self.get_rmse_error,6)))
        plt.show()

    @property
    def get_points(self):
        return np.concatenate([[self.xp],[self.yp]],axis = 0)
    
    @property
    def get_rmse_error(self):
        idp = np.array([np.searchsorted(self.xp,xx,'right') for xx in self.x])-1
        idp[-1] = idp[-1]-1
        return np.square(self.yp[idp]-self.y).mean()**0.5
        
        