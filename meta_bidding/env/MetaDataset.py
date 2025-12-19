from pydoc import resolve
import numpy as np
import pandas as pd
import torch
from typing import *
from collections import *

import os
from zmq import device
import numpy as np
import meta_bidding
from meta_bidding.utils import Batch

# Create an abstract class
class MetaDataset():
    def __init__(self,env_config:Dict):
        raise NotImplementedError

    def reset(self, seed=None):
        raise NotImplementedError

    def mini_batch_step(self, actions):
        raise NotImplementedError

    def step(self, act):
        raise NotImplementedError

    def get_step_reward(self, act):
        raise NotImplementedError

    def get_step_profit(self, act):
        raise NotImplementedError

    def get_minibatch_obs(self):
        raise NotImplementedError

    def get_mini_batch_history(self):
        raise NotImplementedError
    
    def _11201(self,action_in_pm_1):
        """Utility function for transforming action in space [-1,1] to space [0,1]

        :param action_in_pm_1: action in space [-1,1]
        :type action_in_pm_1: float
        """
        return action_in_pm_1/2 + .5

class MetaDatasetAEMO(MetaDataset):
    def __init__(self,env_config:Dict = {'num_agents':1,'device':0,'data_source':'train'}):
        
        self.data_source = env_config['data_source']
        self.num_agents = env_config['num_agents']
        self.parse_data(env_config)
        
        self.device = 'cuda:'+str(env_config['device']) if type(env_config['device']) is int else env_config['device']
        
        self.env_config = env_config
        self.reset() # will define self._pcs and self._soc

    def parse_data(self,env_config):

        data_paths = {
            "train": "meta_bidding/data/aemo_data/aemo_price_train_dual.pkl",
            "test": "meta_bidding/data/aemo_data/aemo_price_test_dual.pkl",
        }

        self.market_keys_rt = [
            'RRP',
            'RAISEREGRRP',
            'LOWERREGRRP',
            'RAISE6SECRRP',
            'RAISE60SECRRP',
            'RAISE5MINRRP',
            'LOWER6SECRRP',
            'LOWER60SECRRP',
            'LOWER5MINRRP',
        ]
        self.market_keys_da = ['DA_' + key for key in self.market_keys_rt]
        self.market_keys = self.market_keys_rt # Compatibility

        # Load Data
        module_root_path = os.path.dirname(os.path.dirname(os.path.abspath(meta_bidding.__file__)))
        self.df_data = pd.read_pickle(os.path.join(module_root_path,data_paths[self.data_source]))
        if 'iso' in env_config:
            print("The AEMO dataset does not have iso information, IGNORING")
        if 'node' in env_config: # df.node in env_config['node']
            self.df_data = self.df_data[self.df_data.REGIONID.isin(env_config['node'])]
            print(f"Setting node : {env_config['node']}")
        if self.df_data.shape[0] == 0:
            raise ValueError("No data in the given iso and node")
        
        # mask prices for markets
        self.energy_market_valid = 1.0
        self.regulation_market_valid = 1.0
        self.reserve_market_valid = 1.0
        if 'energy' not in env_config['product']:
            self.energy_market_valid = 0.0
        if 'regulation' not in env_config['product']:
            self.regulation_market_valid = 0.0
        if 'reserve' not in env_config['product']:
            self.reserve_market_valid = 0.0
        print(f"Participating {len(env_config['product'])} market: {env_config['product']}")
        
        # load mean std information
        mean_std_data_path = os.path.join(module_root_path,data_paths["train"])
        df_train = pd.read_pickle(mean_std_data_path)
        self.price_mean = df_train[self.market_keys_rt].mean()
        self.price_std = df_train[self.market_keys_rt].std()
        

        # numpify the price data
        # A. Process Real-Time Data (RT)
        self._lmp_rt = self.df_data[self.market_keys_rt].to_numpy()
        self._lmp_rt_normalized = (self.df_data[self.market_keys_rt]-self.price_mean)/self.price_std
        self._lmp_rt_normalized = self._lmp_rt_normalized.to_numpy()

        # B. Process Day-Ahead Data (DA)
        if set(self.market_keys_da).issubset(self.df_data.columns):
            self._lmp_da = self.df_data[self.market_keys_da].to_numpy()
            # IMPORTANT: Normalize DA using RT stats to preserve price spread
            self._lmp_da_normalized = (self.df_data[self.market_keys_da].values - self.price_mean.values) / self.price_std.values
        else:
            print("\033[91mWARNING: Day-Ahead (DA) data not found. Fallback: Copying RT data to DA.\033[0m")
            self._lmp_da = self._lmp_rt.copy()
            self._lmp_da_normalized = self._lmp_rt_normalized.copy()

        # C. Compatibility Aliases
        self._lmp = self._lmp_rt
        self._lmp_normalized = self._lmp_rt_normalized

        # Generate normalized price range for sampling supply curves
        self.M = 256
        self.quantile_range = np.concatenate([np.linspace(0.001,0.004,self.M//4),
                                              np.linspace(0.004,0.1,self.M//4),
                                              np.linspace(0.1,0.5,self.M//4),
                                            np.linspace(0.5**3,0.99,self.M//4)**(1/3)])
        # self.reg_quantile_range = np.concatenate([np.linspace(0.005,0.05,self.M//2),
        #                                       np.linspace(0.05,0.5,self.M//4),
        #                                     np.linspace(0.5**3,0.99,self.M//4)**(1/3)]).reshape(-1,1).repeat(2,axis = 1)
        self.price_range = self.df_data[self.market_keys].quantile(self.quantile_range).to_numpy() 
        self.price_range[:,0] = np.concatenate(
            [
                np.linspace(-50,0,self.M//4),
                np.linspace(0,100,self.M//4),
                np.linspace(100,250,self.M//8),
                np.linspace(250,600,self.M//4),
                np.linspace(600,2000,self.M//8)
            ]
        )
        
        # self.price_range[:,1:] =  np.concatenate(
        #     [
        #         np.linspace(-0.1,30,self.M//2),
        #         np.linspace(30,100,self.M//4),
        #         np.linspace(100,400,self.M//4),
        #     ]
        # ).reshape(-1,1).repeat(8,axis = 1)
        self.price_range[:,1:] = np.linspace(-0.1,50,self.M).reshape(-1,1).repeat(8,axis = 1)

        self.price_range_normalized = (self.price_range-self.price_mean.to_numpy())/self.price_std.to_numpy()


        # # TEMP for debug # print color red "Warning, Energy only markey for debug"
        # print("\033[91mWARNING: Energy and Regulation only market for debug")
        # self._lmp[:,3:] = 0
        # self._lmp_normalized[:,3:] = 0

        # create indexes fro inputs
        self.encoder_lmp_shift = np.tile(np.arange(-288 * 4, 0), (self.num_agents, 1))
        self.encoder_fix_embedding = np.tile(np.eye(9), (self.num_agents, 1, 1))

        self.decoder_lmp_shift = np.tile(0, (self.num_agents, 1))
        # self.decoder_fix_embedding = np.tile(np.eye(9), (self.num_agents, 1, 1))

        # create observations of mean and std
        self.hour_price_mean = self._lmp_normalized.reshape(-1,12,9).mean(axis = 1)
        self.hour_price_std = self._lmp_normalized.reshape(-1,12,9).std(axis = 1)
        self.hour_hist_idx_from_pc = lambda pc: (pc/12).astype(int)[:,None] + np.arange(-24*4,0)[None,:]
        self.hour_inday_hist_idx_from_pc = lambda pc: (pc//12).astype(int)[:,None] + np.arange(-6,0)[None,:] # filter the last 6 hours of the day
 

        # Timestamp embeddings
        time_stamp_timeofday = self.df_data.SETTLEMENTDATE.dt.time
        timeofday_seconds = time_stamp_timeofday.apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
        # bin the value to 0 and 2pi
        timeofday_seconds = timeofday_seconds/86400*2*np.pi
        # compute the positional encoding
        self.positional_encoding_timeofday = np.array([np.sin(timeofday_seconds), np.cos(timeofday_seconds)]).T
        
        self.timeofday_shift = np.zeros((self.num_agents,9),dtype = np.int64)

        # Construct the the Start Of Day index lists
        self.sod_idx = np.where((timeofday_seconds==0).values)[0]
        
        # Create Auxiliary Storage states
        self.dataset_size = self._lmp_normalized.shape[0]
        self.observation_size = self.decoder_lmp_shift # + self.decoder_fix_embedding.shape[-1]

        # SoC violation loss parameters
        self.desired_soc_violation_freq = 0.03
        self.beta_range = [0.02]#[0.1,0.05,0.025,0.01]
        self.beta_id = 0
        self.soc_violation_update_weight = 5e-2
        self.current_soc_violation_freq = 0.04
        self.update_soc_violation_loss()
        
    def update_soc_violation_loss(self):
        if self.current_soc_violation_freq >= self.desired_soc_violation_freq+0.01:
            self.beta_id = max(self.beta_id-1,0)
        elif self.current_soc_violation_freq <= self.desired_soc_violation_freq-0.01:
            self.beta_id = min(self.beta_id+1,len(self.beta_range)-1)
        self.beta = self.beta_range[self.beta_id]
        return
        

    def reset(self, seed=None, MAXSOC = None, DEGRATIO = None, EFFICIENCY = None):
        
        # reset ESS parameters:
        self.MAXP = 1 # x MW
        shape = (self.num_agents,)
        # https://www.nrel.gov/docs/fy23osti/85878.pdf 大部分BESS是4hr
        self.MAXSOC = np.random.uniform(low=self.env_config['soc'], high=self.env_config['soc'], size=shape) if MAXSOC is None else np.array(MAXSOC).reshape(-1)# x MWH
        self.DEGRATIO = np.random.uniform(low=50, high=50, size=shape) if DEGRATIO is None else np.array(DEGRATIO).reshape(-1) # $/MWH-cycle
        self.EFFICIENCY = np.random.uniform(low=np.sqrt(0.9), high=np.sqrt(0.9), size=shape) if EFFICIENCY is None else np.array(EFFICIENCY).reshape(-1)# single direction efficiency
        self.MAXPRTRATIO = self.MAXP/(self.MAXSOC)/12.# percentage can be changed in SoC with Maximun Power(without considering efficiency)
        self.PARAMS = np.stack([self.MAXSOC/10.,self.DEGRATIO/10.,self.EFFICIENCY,self.MAXPRTRATIO]).T
        self.param_dim = self.PARAMS.shape[-1]


        # convert all to torch tensors
        self.MAXSOC = torch.Tensor(self.MAXSOC).to(self.device)
        self.DEGRATIO = torch.Tensor(self.DEGRATIO).to(self.device)
        self.EFFICIENCY = torch.Tensor(self.EFFICIENCY).to(self.device)
        self.MAXPRTRATIO = torch.Tensor(self.MAXPRTRATIO).to(self.device)
        self.PARAMS = torch.Tensor(self.PARAMS).to(self.device)

        self._pcs = np.random.choice(self.sod_idx, size = self.num_agents)
        if hasattr(self, "_soc") and len(self._soc)>1:
            soc_numpy = np.array([s.detach().cpu().numpy() for s in self._soc])
            reset_sample_violation_freq = np.mean(soc_numpy>1) + np.mean(soc_numpy<0)
            self.current_soc_violation_freq = self.current_soc_violation_freq*(1-self.soc_violation_update_weight)\
                                                + self.soc_violation_update_weight*reset_sample_violation_freq
            self.update_soc_violation_loss()

        self._soc = [torch.rand(self.num_agents).to(self.device)]
        mcp_estimate = np.quantile(self._lmp[self._pcs.reshape(-1,1) + np.arange(-288,0),0],0.2,axis = 1)
        self._mcp = torch.Tensor(mcp_estimate).to(self.device)
        return 
    
    def get_action(self, actions, verbose_profit = False):
    
        p_max = torch.tensor([1]).to(self.device)
        p_min = torch.tensor([-1]).to(self.device)

        energy_action = actions[:,0] * self.energy_market_valid
        # Clip Energy Actiong to valid value in testing
        if verbose_profit: 
            energy_action  = torch.clamp(energy_action,-(1-self._soc[-1])/self.MAXPRTRATIO/self.EFFICIENCY,self._soc[-1]/self.MAXPRTRATIO*self.EFFICIENCY)

        actions01 = self._11201(actions)
        regup_action  =  actions01[:,1].clip(None,p_max-energy_action) * self.regulation_market_valid
        regdown_action = actions01[:,2].clip(None,energy_action-p_min) * self.regulation_market_valid
        res6sup_action = actions01[:,3].clip(None,p_max-energy_action-regup_action) * self.reserve_market_valid
        res60sup_action = actions01[:,4].clip(None,p_max-energy_action-regup_action-res6sup_action) * self.reserve_market_valid
        res5minup_action = actions01[:,5].clip(None,p_max-energy_action-regup_action-res6sup_action-res60sup_action) * self.reserve_market_valid
        res6sdown_action = actions01[:,6].clip(None,energy_action-p_min-regdown_action) * self.reserve_market_valid
        res60sdown_action = actions01[:,7].clip(None,energy_action-p_min-regdown_action-res6sdown_action) * self.reserve_market_valid
        res5mindown_action = actions01[:,8].clip(None,energy_action-p_min-regdown_action-res6sdown_action-res60sdown_action) * self.reserve_market_valid

        return energy_action, regup_action, regdown_action, res6sup_action, res60sup_action, res5minup_action, res6sdown_action, res60sdown_action, res5mindown_action
    
    def get_action_hdb(self, price_bids, power_bids):

        """
        Solve power market optimization
        """
        
        power_base = power_bids[0] # note that energy market starts at p_min, AS market starts at 0
        power_bid_segments = power_bids - np.concatenate([[[-1]] + 8*[[0]],power_bids[:,:-1]],axis = 1)

        lmp = self._lmp[self._pcs].flatten() # shape: (9)
        soc = self._soc[-1].detach().cpu().numpy().item() # shape: (9)
        p_max = 1
        p_min = -1
        soc_max = 1
        soc_min = 0
        MAXPRTRATIO = self.MAXPRTRATIO.detach().cpu().numpy().item()
        EFFICIENCY = self.EFFICIENCY.detach().cpu().numpy().item()

        import gurobipy as gp

        # Create a new model
        m = gp.Model("multi-market-joint-clearing")
        m.setParam('OutputFlag', 0)

        # enable markets
        enabled_markets = np.array([
            self.energy_market_valid,
            self.regulation_market_valid,
            self.regulation_market_valid,
            self.reserve_market_valid,
            self.reserve_market_valid,
            self.reserve_market_valid,
            self.reserve_market_valid,
            self.reserve_market_valid,
            self.reserve_market_valid,
        ]).reshape(-1,1).repeat(10,axis =1)

        # Create variables
        p = m.addVars(9, 10, lb=0, ub=power_bid_segments*enabled_markets, name="p")
        p_m = m.addVars(9, lb=-gp.GRB.INFINITY, name="p_m")
        p_positive = m.addVar(lb=-gp.GRB.INFINITY, ub = p_max, name="p_positive") # (1d)
        p_negative = m.addVar(lb= p_min, ub = gp.GRB.INFINITY, name="p_negative") # (1d)
        soc_positive = m.addVar(lb=soc_min, ub=soc_max, name="soc_positive") # (1g)
        soc_negative = m.addVar(lb=soc_min, ub=soc_max, name="soc_negative") # (1g)

        

        # Set objective
        m.setObjective(
            gp.quicksum(
                p[m,n]*(lmp[m] - price_bids[m,n]) 
                for m in range(9) 
                for n in range(10)
            ), 
            gp.GRB.MAXIMIZE
        )
        
        # Add constraints
        # (1a) p is constrained by power_bid_segments: p <= power_bid_segments
        
        # (1b)(1c) sum of p on the dimension of N is p_m
        _1b = m.addConstrs((gp.quicksum(p[m,n] for n in range(10)) == p_m[m] for m in range(1,9)), "1b")
        _1c = m.addConstr(gp.quicksum(p[0,n] for n in range(10)) + p_min*self.energy_market_valid == p_m[0], "1c")

        # (1d)-(1f) p+ p- is constrained by p_max and p_min (1d already satisfied)
        # p_positive = p_m[0] + p_m[1] + p_m[3] + p_m[4] + p_m[5]
        _1e = m.addConstr((p_positive == gp.quicksum(p_m[m] for m in [0,1,3,4,5])), "1e")
        # p_negative = p_m[0] - p_m[2] - p_m[6] - p_m[7] - p_m[8]
        _1f = m.addConstr((p_negative == p_m[0] - gp.quicksum(p_m[m] for m in [2,6,7,8])), "1f")

        # (1g)-(1i) SoC Constraints (1g already satisfied)
        # soc_positive = soc + p_positive * MAXPRTRATIO / EFFICIENCY
        _1h = m.addConstr((soc_positive == soc - (p_m[0]+0.25*p_m[1]-0.25*p_m[2]) * MAXPRTRATIO / EFFICIENCY), "1h")
        # soc_negative = soc + p_negative * MAXPRTRATIO * EFFICIENCY
        _1i = m.addConstr((soc_negative == soc - (p_m[0]+0.25*p_m[1]-0.25*p_m[2]) * MAXPRTRATIO * EFFICIENCY), "1i")

        # solve the optimization
        m.optimize()
        
        # if infeasible
        if m.status == gp.GRB.INFEASIBLE:
            import ipdb;ipdb.set_trace()

        # Retrieve the results
        return np.array([p_m[m].x for m in range(9)])



    
    def mini_batch_step(self, action_rt, action_da=None, verbose_profit = False):
        """
            Run 24*1h steps in the environment
            input: action_rt: (batch_size, 9), type: torch.tensor (Raw Actions)
                   action_da: (batch_size, 9), type: torch.tensor (Raw Actions) or None
            output: soc: 24h*(batch_size, 1), type: List[torch.tensor]
                    rew: 24h*(batch_size, 1), type: List[torch.tensor]
        """
        # 1. Retrieve Prices
        lmps_rt_numpy = self._lmp_rt[self._pcs]
        lmps_rt = torch.tensor(lmps_rt_numpy, dtype=torch.float32).to(self.device) # shape: (9, batchsize)
        
        lmps_da_numpy = self._lmp_da[self._pcs]
        lmps_da = torch.tensor(lmps_da_numpy, dtype=torch.float32).to(self.device)

        # 2. Action Processing (Raw -> Physical)
        # RT Actions (Physical Reality)
        (energy_action_rt, regup_action_rt, regdown_action_rt, 
         res6sup_action_rt, res60sup_action_rt, res5minup_action_rt, 
         res6sdown_action_rt, res60sdown_action_rt, res5mindown_action_rt) = self.get_action(action_rt, verbose_profit=verbose_profit)

        # DA Actions (Financial Commitment)
        if action_da is not None:
            # DA actions are not constrained by RT SoC, so verbose_profit=False
            (energy_action_da, regup_action_da, regdown_action_da, 
             res6sup_action_da, res60sup_action_da, res5minup_action_da, 
             res6sdown_action_da, res60sdown_action_da, res5mindown_action_da) = self.get_action(action_da, verbose_profit=False)

        # 3. Physics Update (Based ONLY on RT)
        # Change Step SoC
        discharge_soc_action = energy_action_rt*(energy_action_rt>0) + 0.25*regup_action_rt/self.EFFICIENCY
        charge_soc_action = energy_action_rt*(energy_action_rt<=0) - 0.25*regdown_action_rt/self.EFFICIENCY
        total_soc_discharge_action = discharge_soc_action + charge_soc_action
        new_soc = self._soc[-1]-self.MAXPRTRATIO*total_soc_discharge_action
        self._soc.append(new_soc)

        # Calculate the mean charging price (RT)
        with torch.no_grad():
            clipped_ori_soc = torch.clamp(self._soc[-2],0,1)
            clipped_new_soc =torch.clamp(self._soc[-1],0,1)
            energy_lmp = lmps_rt[:,0]            
            self._mcp = (total_soc_discharge_action>=0) * self._mcp +\
                        (total_soc_discharge_action<0)*(clipped_ori_soc*self._mcp+(clipped_new_soc-clipped_ori_soc)*energy_lmp)/(clipped_new_soc+1e-7)
            self._mcp = torch.clip(self._mcp,-200,500)
            

        # Compute Energy Reward Efftiveness Ratio
        energy_rew_discount = 1 - torch.sigmoid((new_soc-1)*6/self.beta) * (new_soc>(1-self.beta))\
                                - torch.sigmoid(-new_soc*6/self.beta) * (new_soc<self.beta)
        
        # 4. Financial Settlement (Two-Stage)
        # Helper for settlement: R = Q_da * P_da + (Q_rt - Q_da) * P_rt
        def settlement(q_rt, q_da, p_rt, p_da):
            if q_da is None:
                return q_rt * p_rt
            return q_da * p_da + (q_rt - q_da) * p_rt

        # Energy Market
        # Base Energy
        rev_energy_base = settlement(energy_action_rt, energy_action_da if action_da is not None else None, lmps_rt[:,0], lmps_da[:,0])
        # Regulation Energy (RT only)
        rev_reg_energy = (0.25*regup_action_rt*lmps_rt[:,0] - 0.25*regdown_action_rt*lmps_rt[:,0])
        
        energy_reward = rev_energy_base + rev_reg_energy

        if not verbose_profit: # Training
            energy_reward = energy_reward * energy_rew_discount
        
        # AS Markets
        reward_market_revenue = energy_reward
        
        # List of AS components for iteration
        as_rt = [regup_action_rt, regdown_action_rt, res6sup_action_rt, res60sup_action_rt, 
                 res5minup_action_rt, res6sdown_action_rt, res60sdown_action_rt, res5mindown_action_rt]
        
        if action_da is not None:
            as_da = [regup_action_da, regdown_action_da, res6sup_action_da, res60sup_action_da, 
                     res5minup_action_da, res6sdown_action_da, res60sdown_action_da, res5mindown_action_da]
        else:
            as_da = [None] * 8

        for i in range(8):
            # Market index i+1
            reward_market_revenue += settlement(as_rt[i], as_da[i], lmps_rt[:,i+1], lmps_da[:,i+1])

        reward_market_revenue = reward_market_revenue * self.MAXP
        
        # Degradation (RT)
        reward_degradation = - self.DEGRATIO*self.MAXP*(energy_action_rt*(energy_action_rt>0) + 0.25*regup_action_rt)
        
        # SoC Violation (RT)
        reward_soc_equivalent_price = - (50/self.beta**2) * (new_soc - (1-self.beta))**2 * (new_soc>(1-self.beta))\
                                    - (50/self.beta**2) * (new_soc - self.beta)**2 * (new_soc<(self.beta))
        reward_soc_violation = reward_soc_equivalent_price * self.MAXP
        
        # --- Deviation Penalty ---
        reward_deviation_penalty = 0
        if action_da is not None:
            # 1. Penalty Rate: 1.5 * Mean RT Price (Global Mean)
            penalty_rate = 1.5 * torch.tensor(self.price_mean.values, device=self.device, dtype=torch.float32)
            
            # 2. Deviation Magnitude: |RT - DA|
            # Energy
            deviation_abs = torch.abs(energy_action_rt - energy_action_da) * penalty_rate[0]
            # AS Markets
            for i in range(8):
                deviation_abs += torch.abs(as_rt[i] - as_da[i]) * penalty_rate[i+1]
            
            # 3. Calculate Penalty (Negative Reward)
            reward_deviation_penalty = -1.0 * deviation_abs * self.MAXP

        if not verbose_profit: # Training
            rew = reward_market_revenue + reward_degradation + reward_soc_violation + reward_deviation_penalty
        else: # testingp_max
            rew = reward_market_revenue + reward_degradation + reward_deviation_penalty

        self._pcs = (self._pcs+1)%self.dataset_size

        info = None
        if verbose_profit:
            info = {
                'rew_soc_violation':(reward_soc_violation + energy_reward*(energy_rew_discount-1)).cpu().numpy(),
                'lmp_da': lmps_da_numpy,
            }
            if action_da is not None:
                # Calculate DA Revenue for stats
                rev_da = energy_action_da * lmps_da[:,0]
                for i in range(8):
                    rev_da += as_da[i] * lmps_da[:,i+1]
                rev_da = rev_da * self.MAXP
                
                info['rev_da'] = rev_da.cpu().numpy()
                info['rev_total'] = reward_market_revenue.cpu().numpy()
                info['rev_rt_deviation'] = info['rev_total'] - info['rev_da']
                info['penalty_dev'] = reward_deviation_penalty.cpu().numpy()

        if not verbose_profit:
            return self._soc[-1],rew,lmps_rt_numpy, None
        else:
            return self._soc[-1],rew,lmps_rt_numpy, info

    def get_minibatch_obs_rnn(self):
        """"
            Get the observation without splitting different markets/hours
            output: obs: (batch_size, obs_dim) type torch.tensor
        """

        return torch.Tensor(np.concatenate([
            self._lmp_normalized[self._pcs],
            self.positional_encoding_timeofday[self._pcs],
            ], axis = -1)).to(self.device)


    def get_minibatch_obs(self):
        """
            Get the observation of the current state
            output: obs: (batch_size, 9, obs_dim), type: torch.tensor
        """
        return torch.Tensor(np.concatenate([
                                self._lmp_normalized[self._pcs.reshape(-1,1)+self.decoder_lmp_shift].swapaxes(-1,-2), # lmp input
                                self.positional_encoding_timeofday[self._pcs[:,None]+self.timeofday_shift]
                                ],axis = -1)).to(self.device)
    
    def get_minibatch_obs_HDB(self):
        """
            Get an observation to sample out the Supply Curve for generating HDBs
            output: obs: (M, 9, obs_dim), type: torch.tensor
        """
        assert self._pcs.shape[0] == 1, "Only support single agent"
        return torch.Tensor(np.concatenate([self.price_range_normalized.reshape(-1,9,1), # lmp input
                                            self.positional_encoding_timeofday[self._pcs[:,None]+self.timeofday_shift].repeat(self.M,axis = 0)
                                            ],axis = -1)).to(self.device)
    
    def get_mini_batch_history(self):
        """
            Get the history of the past seven days
            output: obs: (batch_size, num_day, 24, obs_dim), type: torch.tensor
        """
        
        return torch.Tensor(
            np.concatenate([self._lmp_normalized[self._pcs.reshape(-1,1)+self.encoder_lmp_shift].swapaxes(-1,-2), # lmp input,
            self.encoder_fix_embedding, # market type input
            ],axis = -1)).to(self.device)
    
    def get_hour_hist(self):
        H = np.concatenate([
            self.hour_price_mean[self.hour_hist_idx_from_pc(self._pcs)],
            self.hour_price_std[self.hour_hist_idx_from_pc(self._pcs)]
            ],axis = -1
        ).swapaxes(-1,-2)
        return  torch.tensor(H,dtype = torch.float32).to(self.device)
    
    def get_hour_inday_hist(self):        
        H = self._lmp_normalized[self._pcs.reshape(-1,1) + np.arange(-72,-0)].swapaxes(-1,-2)
        return torch.tensor(H,dtype = torch.float32).to(self.device)
    
    def get_HDB(self, supply_curves):
        price_range = self.price_range
        price_bids = []
        power_bids = []
        for k in range(9):
            price_bid, power_bid = self._get_HDB2(price_range[:,k],supply_curves[:,k])
            price_bids.append(price_bid)
            power_bids.append(power_bid)
        if (np.array(power_bids[1:])<0).any():
            print("Warning: Negative Power Bids")
            for k in range(1,9):
                power_bids[k] = np.clip(power_bids[k],0,None)

        # # manage oppurtunity cost of regulation market
        # lmp_1hr = self._lmp[self._pcs - np.arange(12)]
        # FR_OC = np.mean(lmp_1hr[:,[3,4,5]],axis = 0).max()
        # RL_OC = np.mean(lmp_1hr[:,[6,7,8]],axis = 0).max()
        # price_bids[1] = price_bids[1] - FR_OC
        # price_bids[2] = price_bids[2] - RL_OC

        return np.array(price_bids), np.array(power_bids)

    def _get_HDB(self,price_,power_):
        """
            Solve the HDB for the given supply curve
        """
        from meta_bidding.utils.hdb import step_solver
        solver = step_solver(price_,power_)
        [solver.step() for _ in range(10)]
        bid_pairs = (solver.xp[1:-1],solver.yp[1:-1])
        return bid_pairs
    
    def _get_HDB2(self,price_,power_):
        """
            Solve the HDB for the given supply curve
        """
        # price_split = np.array_split(price_, 10)
        # power_split = np.array_split(power_, 10)
        # price_bid = np.array([pr.mean() for pr in price_split])
        # power_bid = np.array([pw[-1] for pw in power_split])
        
        # patch start and end
        power_[0] = -(power_<0).any().astype('float')

        # return (price_bid, power_bid)
        power_diff = np.diff(power_)/(power_[-1] - power_[0] + 0.01)
        price_diff = np.diff(price_)/(price_[-1] - price_[0])
        diff = np.sqrt(power_diff**2 + np.mean(price_diff)**2)
        # np.mean(power_diff)#
        sum_diff = np.cumsum(diff)
        anchor_sum = np.linspace(0,1,11)[1:] * sum_diff[-1]
        anchor_idx = np.searchsorted(sum_diff,anchor_sum,'left')
        # make unique and remove zero
        anchor_idx = np.unique(anchor_idx)
        anchor_idx = anchor_idx[anchor_idx>0]

        price_split = np.array_split(price_, anchor_idx)
        power_split = np.array_split(power_, anchor_idx)

        price_bid = np.array([pr[-1] for pr in price_split[:-1]])
        power_bid = np.array([pw.mean() for pw in power_split[1:]])

        if len(price_bid)!=10 and len(price_bid)>0:
            # replicate the last bid
            num_bid_add = 10 - len(price_bid)
            price_bid = np.concatenate([price_bid,np.repeat(price_bid[-1],num_bid_add)])
            power_bid = np.concatenate([power_bid,np.repeat(power_bid[-1],num_bid_add)])
        elif len(price_bid)==0:
            price_bid = np.repeat(price_[0],10)
            power_bid = np.repeat(power_[0],10)

        # force the last bid to be full output
        power_bid[-1] = 1
        return (price_bid, power_bid)


    
    def _get_terminal_rew(self):
        # the rest of the power will be sold in the energy market
        soc = self._soc[-1]
        lmp = self._lmp[self._pcs.reshape(-1,1) + np.arange(288),0]
        lmp_quantile = np.quantile(lmp,1-self.MAXSOC[0].cpu().numpy()/24.,axis = 1)
        lmp_quantile = torch.tensor(lmp_quantile).to(self.device)
        
        # predict the average step profit of the rest of the SoC is the energy is sold in the NEXTDAY energy market
        predicted_rew = soc * self.MAXSOC * lmp_quantile * self.EFFICIENCY / 288
        return predicted_rew