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
        
        # 1. rollout one episode
        lmp,soc,action,profit = [],[],[],[]
        for step in range(self.seq_len):
            info = self.one_minibatch_step(verbose=True)
            lmp.append(info['lmp'])
            soc.append(info['soc'])
            action.append(info['action'])
            profit.append(info['reward'])

        # -1. return the reward of the episode (dummy)
        return np.mean(profit)

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

        
        # input shape (batch_size, 18, 96)
        self.lstm_encoder1 = nn.LSTM(input_size=18, 
                        hidden_size=128, 
                        num_layers=1, 
                        batch_first=True,
                        bidirectional=False).to(self.device) # output shape: (batch_size, 24)
        self.lstm_encoder1_compress = nn.Sequential(
            nn.Linear(128,6),
            nn.ReLU()
        ).to(self.device)

        self.lstm_encoder2 = nn.LSTM(input_size=9,
                        hidden_size=64,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False).to(self.device)
        self.lstm_encoder2_compress = nn.Sequential(
            nn.Linear(64,6),
        ).to(self.device)


        self.mlp_decoder = nn.Sequential(
            ClassWiseLinear(9,6+6+4+1,128),
            nn.ReLU(),
            ClassWiseLinear(9,128,256),
            nn.ReLU(),            
            ClassWiseLinear(9,256,128),
            nn.ReLU(),
            ClassWiseLinear(9,128,1),
            nn.Tanh(),
        ).to(self.device)


        self.layers = nn.ModuleList([self.lstm_encoder1,self.lstm_encoder1_compress,self.lstm_encoder2,self.lstm_encoder2_compress,self.mlp_decoder])
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=learning_rate)    


    def one_minibatch_step(self,verbose = False, HDB = False):
        # 1. get actions from H(history) and P(price)
        H = self.env.get_hour_hist().swapaxes(-1,-2)
        encoded_H,_= self.lstm_encoder1(H) # encoded_H.shape = (batch_size, 32)
        encoded_H = self.lstm_encoder1_compress(encoded_H[:,-1:,:]).repeat((1,9,1))

        # get action for each five minutes
        socs, rews, lmps, actions = [],[],[],[]
        for hour in range(24):
            # encode indat hist hourly
            inday_hist = self.env.get_hour_inday_hist().swapaxes(-1,-2)
            encoded_inday_hist,_ = self.lstm_encoder2(inday_hist)
            encoded_inday_hist = self.lstm_encoder2_compress(encoded_inday_hist[:,-1:,:]).repeat((1,9,1))
            known_soc = self.env._soc[-1]# update the soc
            mcp = self.env._mcp/250.
            for t in range(12):
                if not HDB: # Use NNSF for bidding
                    X = self.env.get_minibatch_obs()
                    action_raw = self.mlp_decoder(torch.cat([encoded_H,
                                                            encoded_inday_hist,
                                                            X,
                                                            known_soc.unsqueeze(1).repeat(1,9).unsqueeze(-1),
                                                            mcp.unsqueeze(1).repeat(1,9).unsqueeze(-1)
                                                            ],axis = -1)).squeeze(-1)
                    # 2. update the soc, reward with the bidding actions
                    action = self.env.get_action(action_raw,verbose_profit=verbose)
                    action = torch.stack(action)
                    mono_supply_curves, price_bids, power_bids = None, None, None
                else: # Generate HDB for bidding
                    with torch.no_grad():
                        X = self.env.get_minibatch_obs_HDB()
                        supply_curves = self.mlp_decoder(torch.cat([encoded_H.repeat(self.env.M,1,1),
                                                                    encoded_inday_hist.repeat(self.env.M,1,1),
                                                                    X,
                                                                    known_soc.expand(self.env.M,9,1),
                                                                    mcp.expand(self.env.M,9,1)
                                                                    ],axis = -1)).squeeze(-1)
                        supply_curves = supply_curves.cpu().numpy()
                        supply_curves[:,1:] = supply_curves[:,1:]/2+0.5 # scale the supply curves in ancillary markets
                        mono_supply_curves = np.maximum.accumulate(supply_curves,axis = 0)
                        price_bids, power_bids = self.env.get_HDB(mono_supply_curves) # HDBs of shape (2,9,10) (price+power, markets, bids)
                        action = self.env.get_action_hdb(price_bids, power_bids) # action tensor of shape (9,1)
                        action = torch.tensor(action,device = self.device, dtype = torch.float32).reshape(9,1)
                soc,rew,lmp,info = self.env.mini_batch_step(action,verbose_profit=verbose) # PC+1~
                socs.append(soc)
                rews.append(rew)
                lmps.append(lmp)
                actions.append(action)

        # 4. return the key information of today the next_day observations for bidding
        if not verbose:
            return rews 
        else:
            return {
                'lmp':np.concatenate(lmps, axis=0),
                'soc': torch.stack(socs).cpu().numpy(),
                'action': torch.stack(actions).cpu().numpy(),
                'reward': torch.stack(rews).cpu().numpy(),
                "mono_supply_curves": mono_supply_curves,
                "price_bids": price_bids,
                "power_bids": power_bids,
                }
