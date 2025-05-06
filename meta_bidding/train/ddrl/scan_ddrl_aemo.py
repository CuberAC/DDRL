# %%
import os
import torch
import numpy as np
import pandas as pd
import tqdm
import argparse
import json
import cvxpy as cp
import meta_bidding
from ddrl_trainer import LSTMTrainer
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
device = 'cuda:0'

# %%
class RolloutBuffer:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.obs = []
        self.lmp = []
        self.action = []
        self.reward = []
        self.profit = []
        self.soc = []

    def append(self,obs,lmp,action,reward,profit,soc):
        self.obs.append(obs)
        self.lmp.append(lmp)
        self.action.append(action)
        self.reward.append(reward)
        self.profit.append(profit)
        self.soc.append(soc)
    
    def to_numpy(self):
        self.obs = np.array(self.obs).squeeze()
        self.lmp = np.array(self.lmp)
        self.action = np.array(self.action)
        self.reward = np.array(self.reward).flatten()
        self.profit = np.array(self.profit).flatten()
        self.soc = np.array(self.soc).flatten()

    
    def save(self,file_name="rollout.csv"):
        self.to_numpy()
        df = pd.DataFrame({
            "lmp":self.lmp,
            "action":self.action,
            "reward":self.reward,
            "profit":self.profit,
            "soc":self.soc,
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
        self.profit = df["profit"].values
        self.soc = df["soc"].values
        self.obs = []
        obs_len = len(df.columns) - 7
        for i in range(obs_len):
            self.obs.append(df["obs"+str(i)].values)
        self.obs = np.array(self.obs).T
        print("loaded from " + file_name)

# %%

ckpt_list = [
    "XXXXXXXXXXXX",
    ]


# %%
df = pd.DataFrame(columns=["method","product","node","soc","profit","ckpt"])
for checkpoint_path in ckpt_list:
    dir_path = os.path.dirname(checkpoint_path)  # Extract the directory path
    param_path = os.path.join(dir_path, "param.json") # Extract the param path
    print("loading configurations from: " + param_path)

    with open(param_path, 'r') as f:
        args = argparse.Namespace(**json.load(f))
    for node in args.node:

       

        trainer_dict = {
            "lstm":LSTMTrainer
        }

        TrainerModel = trainer_dict[args.trainer]
        trainer = TrainerModel(batch_size=1,
                            seq_len=None,
                            device=device,
                            env_config= vars(args)|{
                                "data_source":"test",
                                'node':[node],
                            })
        trainer.load(checkpoint_path)


        # %%
        rollout_len = len(trainer.env.df_data)//288
        print(f"Rollout length: {rollout_len}")
        trainer.reset()
        # trainer.env.reset(MAXSOC = np.array([6.]), DEGRATIO = np.array([20.]), EFFICIENCY = np.array([0.9**0.5]))
        trainer.env._pcs =trainer.env._pcs * 0 

        # %%

        buffer = RolloutBuffer()
        lmp,action,soc,profit = np.zeros((rollout_len,)),np.zeros((rollout_len,)),np.zeros((rollout_len,)),np.zeros((rollout_len,))
        for i in tqdm.tqdm(range(rollout_len)):
            with torch.no_grad():
                info = trainer.one_minibatch_step(verbose=True, HDB = True)
            buffer.append(obs = None,
                        lmp = info['lmp'],
                        action = info['action'],
                        reward = info['reward'],
                        profit = info['reward'],
                        soc = info['soc'],
                        )
        buffer.to_numpy()

        buffer.lmp = buffer.lmp.reshape(-1,buffer.lmp.shape[-1])
        buffer.action = buffer.action.reshape(-1,buffer.action.shape[-1])
        buffer.reward = buffer.reward.reshape(-1)
        buffer.profit = buffer.profit.reshape(-1)
        buffer.soc = buffer.soc.reshape(-1)

        print(f"Average Profit: {buffer.profit.mean()}")

        df = df._append({
            "method":"ddrl",
            "product":args.product,
            "node":node,
            "soc":args.soc,
            "profit":buffer.profit.mean(),
            "ckpt":checkpoint_path,
            "vaid_soc":np.mean((buffer.soc>0) * (buffer.soc<1)),
        },ignore_index=True)

        print(buffer.profit.mean())

# save the results
import datetime
df.to_csv(f"aemo_results_{datetime.datetime.now()}.csv",index=False)
