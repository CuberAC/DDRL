import numpy as np
import torch
import meta_bidding
from torch import nn
from meta_bidding.utils.layer import ClassWiseLinear


class DDRLTrainer(nn.Module):

    def __init__(self,batch_size = 64,seq_len = 30,learning_rate = 1e-4,device = 'cuda:0',env_config = {}) -> None:
        super(DDRLTrainer,self).__init__()
        
        # 1.Define the hyperparameters
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.device = device
        self.num_hist_days = 7
        # 1. Initialize the environment
        self.env = meta_bidding.env.MetaDatasetAEMO(
            {'num_agents':self.batch_size,'device':self.device,'data_source':'train'}|env_config)

    def reset(self):
        self.env.reset()
        self.PARAMS = self.env.PARAMS
        self.soc = self.env._soc[-1] # the main purpose of the self.soc is to be a placeholder for state of charges in the environment
        return self.env.get_minibatch_obs()

    def train_eps(self):
        # 0. Initialize the hidden values
        eps_rew = [] # Store the reward of each episode
        loss = 0 # Store the loss of each episode
        self.reset() #  Reset the environment
        
        # 2. rollout one episode
        for step in range(self.seq_len):
            rew = self.one_minibatch_step() # rew: 24h*(batch_size), type: List[torch.tensor]
            rew = torch.stack(rew) # rew: (24h, batch_size), type: torch.tensor
            loss = loss-torch.mean(rew)/self.seq_len
        loss = loss - torch.mean(self.terminal_rew())/self.seq_len
            
        # 3. backpropagate the loss
        eps_rew.append(-loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        
        # --- DEBUG: Check Gradient Flow ---
        da_grad_norm = 0.0
        for name, param in self.da_decoder.named_parameters():
            if param.grad is not None:
                da_grad_norm += param.grad.norm().item()
        
        rt_grad_norm = 0.0
        for name, param in self.rt_decoder.named_parameters():
            if param.grad is not None:
                rt_grad_norm += param.grad.norm().item()
                
        if da_grad_norm == 0.0:
            print(f"[WARNING] DA Decoder Gradient is DOAD (Zero)! RT Grad: {rt_grad_norm:.4f}")
        # else:
        #     print(f"[DEBUG] DA Grad: {da_grad_norm:.4f} | RT Grad: {rt_grad_norm:.4f}")
        # ----------------------------------

        nn.utils.clip_grad_norm_(self.layers.parameters(), max_norm=10, norm_type = 2)
        self.optimizer.step()

        # -1. return the reward of the episode (dummy)
        return np.mean(eps_rew)

    def evaluate_eps(self):
        # 0. Initialize the hidden values
        self.reset() #  Reset the environment
        
        # 1. Initialize logs container
        logs = {
            'lmp': [], 
            'lmp_da': [],
            'soc': [], 
            'rt_action': [], 
            'da_action': [], 
            'reward': [],
            'rev_da': [],
            'rev_rt_deviation': [],
            'rev_total': [],
        }
        
        # 2. rollout one episode
        for step in range(self.seq_len):
            info = self.one_minibatch_step(verbose=True)
            
            # Collect Time-Series Data (Concatenate along time axis later)
            logs['lmp'].append(info['lmp'])
            logs['lmp_da'].append(info['lmp_da'])
            logs['soc'].append(info['soc'])
            logs['rt_action'].append(info['action'])
            logs['reward'].append(info['reward'])
            
            # Collect Daily Data (Stack along day axis later)
            if 'da_action' in info:
                logs['da_action'].append(info['da_action'])
                
            # Collect Financial Stats if available
            if 'rev_da' in info:
                logs['rev_da'].append(info['rev_da'])
                logs['rev_rt_deviation'].append(info['rev_rt_deviation'])
                logs['rev_total'].append(info['rev_total'])

        # 3. Data Aggregation
        results = {}
        
        # Concatenate time-series data (Total Steps = seq_len * 288)
        # Shape: (Total_Steps, Batch, ...)
        results['lmp'] = np.concatenate(logs['lmp'], axis=0)
        results['lmp_da'] = np.concatenate(logs['lmp_da'], axis=0)
        results['soc'] = np.concatenate(logs['soc'], axis=0)
        results['rt_action'] = np.concatenate(logs['rt_action'], axis=0)
        results['reward'] = np.concatenate(logs['reward'], axis=0)
        
        # Stack daily data (Days = seq_len)
        # Shape: (Days, Batch, 9, 24)
        if logs['da_action']:
            results['da_action'] = np.stack(logs['da_action'], axis=0)
            
        # Stack financial stats
        # Shape: (Days, Batch, ...) or (Total_Steps, Batch, ...) depending on source
        # Assuming these are daily or step-wise sums returned by info
        if logs['rev_da']:
             # Note: rev_da in info is likely (Batch, 9) or similar per step/day. 
             # If it's per step, use concatenate. If per day, use stack.
             # Based on MetaDataset logic, rev_da is calculated per step but accumulated? 
             # Actually in MetaDataset it returns step-wise revenue. So concatenate.
             results['rev_da'] = np.concatenate(logs['rev_da'], axis=0)
             results['rev_rt_deviation'] = np.concatenate(logs['rev_rt_deviation'], axis=0)
             results['rev_total'] = np.concatenate(logs['rev_total'], axis=0)

        # Calculate Mean Profit for compatibility
        results['mean_profit'] = np.mean(results['reward'])

        # -1. return the full results dictionary
        return results

    def save(self,save_pth_path):
        """
            Save the model state.dict to path **save_pth_path**
        """
        return torch.save(self.state_dict(),save_pth_path)

    def load(self,load_pth_path):
        """
            Load the model state.dict from path **load_pth_path**
        """
        return self.load_state_dict(torch.load(load_pth_path, map_location=lambda storage, loc: storage))
    
    def terminal_rew(self,):
        """
         unit: rew per step
        """
        return self.env._get_terminal_rew()
    

class LSTMTrainer(DDRLTrainer):
    def __init__(self,batch_size = 64,seq_len = 30,learning_rate = 1e-4,device = 'cuda:0',env_config = {}):
        super(LSTMTrainer,self).__init__(batch_size=batch_size,seq_len=seq_len,learning_rate=learning_rate,device=device,env_config=env_config)
        
        self.num_markets = env_config.get('num_markets', 9)
        self.mode = env_config.get('mode', 'default') # Training mode
        
        # input shape (batch_size, 18, 96)
        # Note: Input size = (mean(num_markets) + std(num_markets)) = 2 * num_markets
        self.lstm_encoder1_input_size = 2 * self.num_markets
        self.lstm_encoder1 = nn.LSTM(input_size=self.lstm_encoder1_input_size, 
                        hidden_size=128, 
                        num_layers=1, 
                        batch_first=True,
                        bidirectional=False).to(self.device) # output shape: (batch_size, 24)
        self.lstm_encoder1_compress = nn.Sequential(
            nn.Linear(128,6),
            nn.ReLU()
        ).to(self.device)

        self.lstm_encoder2 = nn.LSTM(input_size=self.num_markets,
                        hidden_size=64,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False).to(self.device)
        self.lstm_encoder2_compress = nn.Sequential(
            nn.Linear(64,6),
        ).to(self.device)

        # --- New Decoders for Two-Stage Settlement ---
        
        # 1. Day-Ahead Decoder (DA)
        # Input: Long-term History (6) + Short-term History (6) = 12
        # Output: 24 hours of actions for each market
        self.da_decoder = nn.Sequential(
            ClassWiseLinear(self.num_markets, 6+6, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 24), # Output 24 steps at once
            nn.Tanh(),
        ).to(self.device)

        # 2. Real-Time Decoder (RT)
        # Input: History (12) + Current Obs (4) + MCP (1) + DA Commitment (1) = 18
        # Adjusted Input: History(12) + Current Obs (1+2+self.num_markets) + Known SoC(1*num_markets) + MCP(1*num_markets) + DA(1*num_markets)
        
        self.da_decoder = nn.Sequential(
            ClassWiseLinear(self.num_markets, 6+6, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 24), # Output 24 steps at once
            nn.Tanh(),
        ).to(self.device)

        # We will update rt_decoder input size after checking get_minibatch_obs
        # Placeholder for now, assumed dynamic calculation in next steps
        self.rt_decoder_input_dim = 6 + 6 + 1 + 1 + 1 + 2 + 1 # Rough guess: Hist(6)+Inday(6)+LMP(1)+Time(2)+Soc(1)+MCP(1)+DA(1) = 18?
        # If lmp is specific to market, and time is shared...
        
        self.rt_decoder = nn.Sequential(
            ClassWiseLinear(self.num_markets, 6+6+3+1+1+1, 128), # 18
            # Breakdown:
            # Encoded Long Term Hist: 6
            # Encoded Short Term Hist: 6
            # Current Obs (X): 3 (LMP[1] + Time[2])
            # Known SoC: 1
            # MCP: 1
            # DA Action: 1
            # Total: 18
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 256),
            nn.ReLU(),            
            ClassWiseLinear(self.num_markets, 256, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 1),
            nn.Tanh(),
        ).to(self.device)

        # self.mlp_decoder = ... (Deprecated)

        self.layers = nn.ModuleList([
            self.lstm_encoder1,
            self.lstm_encoder1_compress,
            self.lstm_encoder2,
            self.lstm_encoder2_compress,
            self.da_decoder,
            self.rt_decoder
        ])
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=learning_rate)    


    def one_minibatch_step(self,verbose = False, HDB = False):
        # 1. get actions from H(history) and P(price)
        H = self.env.get_hour_hist().swapaxes(-1,-2)
        encoded_H,_= self.lstm_encoder1(H) # encoded_H.shape = (batch_size, 32)
        encoded_H = self.lstm_encoder1_compress(encoded_H[:,-1:,:]).repeat((1,self.num_markets,1))

        # --- New: Generate Day-Ahead Plan ---
        # We need initial short-term history for DA planning
        inday_hist_init = self.env.get_hour_inday_hist().swapaxes(-1,-2)
        encoded_inday_hist_init,_ = self.lstm_encoder2(inday_hist_init)
        encoded_inday_hist_init = self.lstm_encoder2_compress(encoded_inday_hist_init[:,-1:,:]).repeat((1,self.num_markets,1))
        
        # DA Input: Long-term + Initial Short-term
        da_input = torch.cat([encoded_H, encoded_inday_hist_init], dim=-1)
        da_plan_24h_raw = self.da_decoder(da_input) # Shape: (Batch, num_markets, 24)
        
        # [Modified] RT Only Mode: Force DA Action to 0
        if self.mode == 'rt_only':
            da_plan_24h_raw = torch.zeros_like(da_plan_24h_raw)
        
        # --- SoC-aware DA plan (energy市场约束) ---
        # Remvoed no_grad to allow gradient flow for DDRL
        virtual_soc = self.env._soc[-1].detach().clone()  # (B,)
        maxptr_hourly = self.env.MAXPRTRATIO * 12.0  # (B,) Convert 5min ratio to HOURLY ratio
        beta = self.env.beta
        MAXP = self.env.MAXP
        
        da_plans = []
        da_penalties = []

        for h in range(24):
            # 仅约束能源市场（索引0），避免承诺超出可充/可放能力
            # Get raw actions for this hour
            action_raw_h = da_plan_24h_raw[:, :, h] # (B, num_markets)
            energy_da_raw = action_raw_h[:, 0]

            # 改为软约束模式：直接使用原始动作更新 SoC，并计算惩罚
            
            # 1. Update SoC based on raw action
            virtual_soc = virtual_soc - maxptr_hourly * energy_da_raw
            
            # 2. Calculate soft constraint penalty (Match MetaDataset Logic)
            # Penalty = - (50 / beta^2) * MAXP * [ (soc-(1-b))^2 * I(soc>1-b) + (soc-b)^2 * I(soc<b) ]
            
            penalty_high = (virtual_soc - (1-beta))**2 * (virtual_soc > (1-beta))
            penalty_low = (virtual_soc - beta)**2 * (virtual_soc < beta)
            
            # Adjusted penalty coefficient to 1.0 to balance market revenue and constraints
            step_penalty = - (1.0 / (beta**2)) * MAXP * (penalty_high + penalty_low)
            
            da_penalties.append(step_penalty)
            
            # 3. Use raw action for plan (No Clamping)
            da_plans.append(action_raw_h)
            
        # Stack to get (Batch, num_markets, 24)
        da_plan_24h = torch.stack(da_plans, dim=2)
        # ------------------------------------

        # get action for each five minutes
        socs, rews, lmps, actions = [],[],[],[]
        lmps_da = []
        rewards_per_market = []
        rev_das, rev_totals, rev_rt_deviations = [], [], []
        for hour in range(24):
            # Get DA action for this hour
            # Clone to avoid in-place modification error (RuntimeError: ... modified by an inplace operation)
            da_action_current_hour = da_plan_24h[:, :, hour].clone() # Shape: (Batch, num_markets)

            # encode inday hist hourly
            inday_hist = self.env.get_hour_inday_hist().swapaxes(-1,-2)
            encoded_inday_hist,_ = self.lstm_encoder2(inday_hist)
            encoded_inday_hist = self.lstm_encoder2_compress(encoded_inday_hist[:,-1:,:]).repeat((1,self.num_markets,1))
            known_soc = self.env._soc[-1]# update the soc
            mcp = self.env._mcp/250.
            for t in range(12):
                if not HDB: # Use NNSF for bidding
                    X = self.env.get_minibatch_obs()
                    
                    # RT Input Construction
                    # X shape: (Batch, num_markets, 3) where 3 is (LMP + TimeOfDay[2])
                    
                    rt_input = torch.cat([
                        encoded_H, # (B, M, 6)
                        encoded_inday_hist, # (B, M, 6)
                        X, # (B, M, 3)
                        known_soc.unsqueeze(1).repeat(1,self.num_markets).unsqueeze(-1), #(B, M, 1)
                        mcp.unsqueeze(1).repeat(1,self.num_markets).unsqueeze(-1), #(B, M, 1)
                        da_action_current_hour.unsqueeze(-1) #(B, M, 1)
                    ], axis = -1)
                    # Total dims = 6+6+3+1+1+1 = 18
                    
                    action_raw = self.rt_decoder(rt_input).squeeze(-1)
                    action_rt = action_raw
                    mono_supply_curves, price_bids, power_bids = None, None, None
                else: # Generate HDB for bidding
                    with torch.no_grad():
                        X = self.env.get_minibatch_obs_HDB()
                        
                        # RT Input for HDB
                        da_action_expanded = da_action_current_hour.unsqueeze(-1).expand(self.env.M, self.num_markets, 1)
                        
                        rt_input_hdb = torch.cat([
                            encoded_H.repeat(self.env.M,1,1),
                            encoded_inday_hist.repeat(self.env.M,1,1),
                            X,
                            known_soc.expand(self.env.M,self.num_markets,1),
                            mcp.expand(self.env.M,self.num_markets,1),
                            da_action_expanded
                        ], axis = -1)

                        supply_curves = self.rt_decoder(rt_input_hdb).squeeze(-1)
                        supply_curves = supply_curves.cpu().numpy()
                        supply_curves[:,1:] = supply_curves[:,1:]/2+0.5 # scale the supply curves in ancillary markets
                        mono_supply_curves = np.maximum.accumulate(supply_curves,axis = 0)
                        price_bids, power_bids = self.env.get_HDB(mono_supply_curves) # HDBs of shape (2,9,10) (price+power, markets, bids)
                        action_rt_numpy = self.env.get_action_hdb(price_bids, power_bids) # action tensor of shape (9,1)
                        action_rt = torch.tensor(action_rt_numpy, device = self.device, dtype = torch.float32).reshape(self.num_markets,1)
                
                # [Modified] DA Only Mode: Force RT Action = DA Action (Apply to both HDB and Point-Estimate modes)
                # Note: HDB path returns (Num_Markets, 1), need to ensure broadcasting or reshaping if Batch > 1
                # Given current HDB implementation seems specific to Batch=1 or handled internally, we apply override here.
                if self.mode == 'da_only':
                    action_rt = da_action_current_hour

                # Call environment with Two-Stage Settlement
                soc,rew,lmp,info = self.env.mini_batch_step(action_rt, action_da=da_action_current_hour, verbose_profit=verbose) # PC+1~
                
                # [关键修改] 注入日前软约束惩罚
                # 将日前规划阶段计算的惩罚叠加到当前的 Reward 中
                # Average the penalty over the 12 steps of the hour (or apply once?)
                # Apply full penalty per step or distribute? 
                # Ideally, if step_penalty is for the hour, we add it to the hourly reward mass.
                # Here we add it to every step (1/12th) or just add full?
                # The violation happened "at this hour". To be strong, let's add full penalty per step or divide by 12.
                # Given current magnitude (50/beta^2 approx 50/0.0004 = 125,000!), it's HUGE.
                # Let's divide by 12 to spread it over the hour.
                total_step_reward = rew + da_penalties[hour] / 12.0
                
                socs.append(soc)
                rews.append(total_step_reward)
                lmps.append(lmp)
                actions.append(action_rt)
                if verbose:
                    lmps_da.append(info['lmp_da'])
                    if 'reward_per_market' in info:
                        rewards_per_market.append(info['reward_per_market'])
                    if 'rev_da' in info:
                        rev_das.append(info['rev_da'])
                        rev_totals.append(info['rev_total'])
                        if 'rev_rt_deviation' in info:
                            rev_rt_deviations.append(info['rev_rt_deviation'])

        # 4. return the key information of today the next_day observations for bidding
        if not verbose:
            return rews 
        else:
            ret = {
                'lmp':np.stack(lmps, axis=0),
                'lmp_da':np.stack(lmps_da, axis=0),
                'soc': torch.stack(socs).cpu().numpy(),
                'action': torch.stack(actions).cpu().numpy(),
                'reward': torch.stack(rews).cpu().numpy(),
                "mono_supply_curves": mono_supply_curves,
                "price_bids": price_bids,
                "power_bids": power_bids,
                "da_action": da_plan_24h.detach().cpu().numpy()
                }
            if len(rewards_per_market) > 0:
                ret['reward'] = np.stack(rewards_per_market, axis=0)
            if len(rev_das) > 0:
                ret['rev_da'] = np.stack(rev_das, axis=0)
                ret['rev_total'] = np.stack(rev_totals, axis=0)
                if len(rev_rt_deviations) > 0:
                    ret['rev_rt_deviation'] = np.stack(rev_rt_deviations, axis=0)
                else:
                    ret['rev_rt_deviation'] = ret['rev_total'] - ret['rev_da']
            return ret
