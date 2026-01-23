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
        # Wait, let's trace rt_input construction in one_minibatch_step:
        # rt_input = cat([encoded_H(6), encoded_inday_hist(6), X(?), known_soc(1), mcp(1), da_action(1)], axis=-1)
        # X comes from get_minibatch_obs: [lmp(num_markets), positional_encoding(2)] -> size = num_markets + 2
        # So total input size per market?
        # The rt_decoder is ClassWiseLinear(num_markets, input_dim, ...). 
        # The inputs are repeated to have shape (Batch, num_markets, input_dim).
        # encoded_H: (Batch, num_markets, 6)
        # encoded_inday_hist: (Batch, num_markets, 6)
        # X: (Batch, num_markets+2) -> This might be an issue if X is not per-market.
        # Let's check get_minibatch_obs inside MetaDataset.
        
        # rt_input_dim = 6 + 6 + (num_markets + 2) + 1 + 1 + 1 = 17 + num_markets ??
        # In original code with 9 markets: 6+6+4+1+1 = 18?
        # Original X was (Batch, 4): lmp(1? no, lmp is 9), positional(2).
        # Wait, get_minibatch_obs_rnn returns: np.concatenate([self._lmp_normalized[self._pcs], self.positional_encoding_timeofday[self._pcs]], axis=-1)
        # _lmp_normalized is (Batch, 9). pos_enc is (Batch, 2). Total X is (Batch, 11).
        
        # BUT, in one_minibatch_step used "if not HDB: X = self.env.get_minibatch_obs()".
        # Let's see get_minibatch_obs in next turn.
        
        # Assuming for now we fix the decoder size dynamically too.
        # Reviewing one_minibatch_step again:
        # rt_input concat axis=-1.
        # encoded_H: (B, num_markets, 6)
        # encoded_inday_hist: (B, num_markets, 6)
        # X: (B, num_markets + 2) -> This is broadcasted? No, it's (B, obs_dim).
        # To concatenate with (B, num_markets, ...), X needs to be (B, num_markets, ...).
        # In original code: X was probably reshaped or something?
        # Let's check one_minibatch_step carefully.
        
        # Original: rt_input = torch.cat([..., X, ...])
        # If X is (B, 11), and others are (B, 9, 6), this cat would fail unless X is unsqueezed and repeated OR specific dims match.
        # Actually X was just X. The ClassWiseLinear expects (Batch, Class, In_Features).
        # So ALL inputs must be (Batch, num_markets, something).
        # If X is (Batch, num_markets+2), it cannot be simply concatenated to (Batch, num_markets, 6) along last dim?
        # No, it must mean X is treated as features common to all markets? No ClassWiseLinear logic handles "Class" dimension.
        
        # Let's look at ClassWiseLinear input requirement.
        # If input is (Batch, Class, Feature), ClassWiseLinear works.
        # So X must be expanded to (Batch, num_markets, Feature).
        # In original code: X = self.env.get_minibatch_obs().
        
        # We need to read get_minibatch_obs to be sure.
        
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
        
        # --- SoC-aware DA plan (energy市场约束) ---
        # Remvoed no_grad to allow gradient flow for DDRL
        virtual_soc = self.env._soc[-1].detach().clone()  # (B,)
        maxptr = self.env.MAXPRTRATIO  # (B,)
        eff = self.env.EFFICIENCY      # (B,)
        
        da_plans = []

        for h in range(24):
            # 仅约束能源市场（索引0），避免承诺超出可充/可放能力
            # Get raw actions for this hour
            action_raw_h = da_plan_24h_raw[:, :, h] # (B, num_markets)
            energy_da_raw = action_raw_h[:, 0]

            # 可放/可充上限（转为[-1,1]尺度下的物理可行范围）
            # Note: We clamp bounds to be non-negative/valid to avoid errors
            max_discharge = (virtual_soc / (maxptr + 1e-8)) * eff  # 正方向上限
            max_charge = ((1 - virtual_soc) / (maxptr + 1e-8)) * eff  # 负方向（充电）上限

            energy_da_clamped = torch.clamp(energy_da_raw, -max_charge, max_discharge)
            
            # Construct constrained action for this hour
            if self.num_markets > 1:
                action_constrained_h = torch.cat([energy_da_clamped.unsqueeze(1), action_raw_h[:, 1:]], dim=1)
            else:
                action_constrained_h = energy_da_clamped.unsqueeze(1)
            
            da_plans.append(action_constrained_h)

            # 前向推演 SoC（仅按 DA 能源动作，忽略调频/备用影响）
            discharge = torch.clamp(energy_da_clamped, min=0.0)
            charge = torch.clamp(energy_da_clamped, max=0.0) / eff
            virtual_soc = virtual_soc - maxptr * (discharge + charge)
            # Ensure virtual_soc stays in bounds for next step calculation
            virtual_soc = torch.clamp(virtual_soc, 0.0, 1.0)
            
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
                
                # Call environment with Two-Stage Settlement
                soc,rew,lmp,info = self.env.mini_batch_step(action_rt, action_da=da_action_current_hour, verbose_profit=verbose) # PC+1~
                
                socs.append(soc)
                rews.append(rew)
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
