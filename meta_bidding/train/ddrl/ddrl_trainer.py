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
        eps_da_pen = [] 
        eps_rt_pen = []
        loss = 0 # Store the loss of each episode
        self.reset() #  Reset the environment
        
        # 2. rollout one episode
        for step in range(self.seq_len):
            rew, da_pen, rt_pen = self.one_minibatch_step() # rew: 24h*(batch_size), type: List[torch.tensor]
            
            # Record penalties
            eps_da_pen.append(torch.stack(da_pen).mean().detach().cpu().item())
            eps_rt_pen.append(torch.stack(rt_pen).mean().detach().cpu().item())

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

        # Save episode-level average penalties for external logging (e.g., wandb)
        self.current_da_penalty = np.mean(eps_da_pen)
        self.current_rt_penalty = np.mean(eps_rt_pen)

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
        # Input: Long-term History (6) + Short-term History (6) + Current Virtual SOC (1) + Time of Day (2) + Safe_Dis(1) + Safe_Chg(1) + Future DA Prices (24) = 41
        # Output: 1 hour of action for each market (Autoregressive step)
        self.da_decoder = nn.Sequential(
            ClassWiseLinear(self.num_markets, 41, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 128),
            nn.ReLU(),
            ClassWiseLinear(self.num_markets, 128, 1),
            nn.Tanh(), 
        ).to(self.device)

        # 2. Real-Time Decoder (RT)
        # Input: History (12) + Current Obs (4) + MCP (1) + DA Commitment (1) = 18
        # Adjusted Input: History(12) + Current Obs (1+2+self.num_markets) + Known SoC(1*num_markets) + MCP(1*num_markets) + DA(1*num_markets)

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

        # --- New: Generate Day-Ahead Plan (Autoregressive Sandbox) ---
        # 1. 获取用于 DA 规划的静态特征（长短历史）
        inday_hist_init = self.env.get_hour_inday_hist().swapaxes(-1,-2)
        encoded_inday_hist_init,_ = self.lstm_encoder2(inday_hist_init)
        encoded_inday_hist_init = self.lstm_encoder2_compress(encoded_inday_hist_init[:,-1:,:]).repeat((1,self.num_markets,1))
        
        # 2. 初始化沙盘推演的起点 SOC
        virtual_soc = self.env._soc[-1].detach().clone()  # (B,)
        maxptr_hourly = self.env.MAXPRTRATIO * 12.0  # (B,) Convert 5min ratio to HOURLY ratio
        beta = self.env.beta
        MAXP = self.env.MAXP
        
        da_plans = []
        da_penalties = []

        # [Oracle Test] 提取未来24小时（每小时1个采样点，跨度12个5min步长）的主能量市场日前电价
        future_24h_idx = (self.env._pcs.reshape(-1, 1) + np.arange(0, 288, 12)) % self.env.dataset_size
        # 提取真实价格并转换为 tensor
        future_24h_prices = torch.tensor(self.env._lmp_da_normalized[future_24h_idx, 0], dtype=torch.float32, device=self.device)
        # 扩充维度以对齐 num_markets: 形状变为 (Batch, num_markets, 24)
        feat_future_24h = future_24h_prices.unsqueeze(1).repeat(1, self.num_markets, 1)

        # 3. 逐小时自回归推演与动作生成
        for h in range(24):
            # 获取当前真实的 batch_size (处理最后一个不完整的 batch)
            curr_b = encoded_H.shape[0]
            
            # [核心修复] 生成当前小时的时间周期编码 (Time of Day Encoding)
            h_sin = np.sin(2 * np.pi * h / 24.0)
            h_cos = np.cos(2 * np.pi * h / 24.0)
            time_feat = torch.tensor([h_sin, h_cos], dtype=torch.float32, device=self.device)
            # 扩展维度对齐: (Batch, num_markets, 2)
            time_feat_expanded = time_feat.view(1, 1, 2).expand(curr_b, self.num_markets, 2)
            
            # 动态构建包含当前 virtual_soc 和 time_feat 的输入特征
            virtual_soc_expanded = virtual_soc.unsqueeze(1).repeat(1, self.num_markets).unsqueeze(-1)

            # --- [新增: 物理信息引导特征 (Physics-Informed Features)] ---
            # 计算如果在 1 小时内不越过 beta 边界，理论上最大允许的充放电动作大小 (0.0 ~ 1.0+)
            # 仅作为输入特征提供给网络，保留 DDRL 软约束学习机制
            max_safe_dis = ((virtual_soc - beta) / maxptr_hourly).clamp(min=0.0)
            max_safe_chg = (((1-beta) - virtual_soc) / maxptr_hourly).clamp(min=0.0)
            
            feat_max_dis = max_safe_dis.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_markets, 1)
            feat_max_chg = max_safe_chg.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_markets, 1)
            
            # 拼接: 6 + 6 + 1 + 2 + 2 + 24 = 41 维
            da_input = torch.cat([
                encoded_H,
                encoded_inday_hist_init,
                virtual_soc_expanded,
                time_feat_expanded,
                feat_max_dis,
                feat_max_chg,
                feat_future_24h,
            ], dim=-1)
            # -------------------------------------------------------------
            
            # 预测当前这 1 个小时的动作
            action_raw_h_unsqueeze = self.da_decoder(da_input) # Shape: (Batch, num_markets, 1)
            action_raw_h = action_raw_h_unsqueeze.squeeze(-1)  # Shape: (Batch, num_markets)
            
            energy_da_raw = action_raw_h[:, 0]
            
            # 提取调频动作（如果有的话，需从 Tanh 的 [-1, 1] 映射到 [0, 1]）
            if self.num_markets > 1:
                regup_da_raw = action_raw_h[:, 1] / 2 + 0.5
                regdown_da_raw = action_raw_h[:, 2] / 2 + 0.5
            else:
                regup_da_raw = torch.zeros_like(energy_da_raw)
                regdown_da_raw = torch.zeros_like(energy_da_raw)

            efficiency = self.env.EFFICIENCY
            # 区分充放电并计入效率损失和辅助服务容量
            discharge_soc_action_da = energy_da_raw * (energy_da_raw > 0) + 0.25 * regup_da_raw / efficiency
            charge_soc_action_da = energy_da_raw * (energy_da_raw <= 0) - 0.25 * regdown_da_raw / efficiency
            total_soc_action_da = discharge_soc_action_da + charge_soc_action_da
            
            # 1. Update SoC based on realistic physical consumption
            virtual_soc = virtual_soc - maxptr_hourly * total_soc_action_da
            
            # 2. Calculate soft constraint penalty (Match MetaDataset Logic)
            penalty_high = (virtual_soc - (1-beta))**2 * (virtual_soc > (1-beta))
            penalty_low = (virtual_soc - beta)**2 * (virtual_soc < beta)
            
            step_penalty = - (1.0 / (beta**2)) * MAXP * (penalty_high + penalty_low)
            
            da_penalties.append(step_penalty)
            
            # 3. Use raw action for plan
            da_plans.append(action_raw_h)
            
        # Stack to get full 24h plan: Shape (Batch, num_markets, 24)
        da_plan_24h = torch.stack(da_plans, dim=2)
        
        # [Modified] RT Only Mode: Force DA Action to 0
        if self.mode == 'rt_only':
            da_plan_24h = torch.zeros_like(da_plan_24h)
        # ------------------------------------

        # get action for each five minutes
        socs, rews, lmps, actions = [],[],[],[]
        rt_penalties = []
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
                
                # --- [新增] 独立计算并记录 RT 的 SOC 物理越限惩罚金额 ---
                rt_penalty_high = (soc - (1-beta))**2 * (soc > (1-beta))
                rt_penalty_low = (soc - beta)**2 * (soc < beta)
                rt_step_penalty = - (50.0 / (beta**2)) * MAXP * (rt_penalty_high + rt_penalty_low)
                rt_penalties.append(rt_step_penalty)
                # ----------------------------------------------------

                # [关键修改] 注入日前软约束惩罚并防止双重扣款
                if self.mode == 'da_only':
                    # DA action is held for 12 RT steps; raw RT penalty can dominate and destabilize gradients.
                    # Refund most of rt_step_penalty so only a small fraction of physical penalty remains.
                    refund_ratio = 1.0
                    total_step_reward = rew - refund_ratio * rt_step_penalty + da_penalties[hour] / 12.0
                else:
                    # 在正常双阶段模式下，DA 动作不影响底层真实 SoC，
                    # 必须额外加上 da_penalties 才能约束 DA 网络的输出。
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
            return rews, da_penalties, rt_penalties
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
            return ret
