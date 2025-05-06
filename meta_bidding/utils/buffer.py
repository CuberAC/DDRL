import pandas as pd
import numpy as np

class RolloutBuffer:
    def __init__(self) -> None:
        self.reset()
    
    def append(self,obs,lmp,action,reward,mcp,profit,soc):
        self.obs.append(obs)
        self.lmp.append(lmp)
        self.action.append(action)
        self.reward.append(reward)
        self.mcp.append(mcp)
        self.profit.append(profit)
        self.soc.append(soc)


    def reset(self):
        self.obs = []
        self.lmp = []
        self.action = []
        self.reward = []
        self.mcp = []
        self.profit = []
        self.soc = []
    
    def to_numpy(self):
        self.obs = np.array(self.obs).squeeze()
        self.lmp = np.array(self.lmp).flatten()
        self.action = np.array(self.action).flatten()
        self.reward = np.array(self.reward).flatten()
        self.mcp = np.array(self.mcp).flatten()
        self.profit = np.array(self.profit).flatten()
        self.soc = np.array(self.soc).flatten()
    
    def save(self,file_name="rollout.csv"):
        self.to_numpy()
        df = pd.DataFrame({
            "lmp":self.lmp,
            "action":self.action,
            "reward":self.reward,
            "mcp":self.mcp,
            "profit":self.profit,
            "soc":self.soc
        })
        # add obs as a multi-dimensional data
        for i in range(self.obs.shape[1]):
            df["obs"+str(i)] = self.obs[:,i]
        df.to_csv(file_name,index=False)
        print("saved to " + file_name)
    
    def load(self,file_name="rollout.csv"):
        df = pd.read_csv(file_name)
        self.lmp = df["lmp"].values
        self.action = df["action"].values
        self.reward = df["reward"].values
        self.mcp = df["mcp"].values
        self.profit = df["profit"].values
        self.soc = df["soc"].values
        self.obs = []
        obs_len = len(df.columns) - 7
        for i in range(obs_len):
            self.obs.append(df["obs"+str(i)].values)
        self.obs = np.array(self.obs).T
        print("loaded from " + file_name)