import os
import json
from datetime import datetime
import wandb
import numpy as np
from ddrl_trainer import LSTMTrainer
import torch
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()

####### Training Hyperparams ######
parser.add_argument("--cuda",default=0)
parser.add_argument("--env_name", default="metabidding-ddrl")
parser.add_argument("--exp_name", default="meta-ddrl-default")

parser.add_argument("--batch_size",default=128,type=int,help="num of parrallel environments")
parser.add_argument("--total_epoches",default=3000,type=int,help="number of epoches trained")
parser.add_argument("--eps_len",default=4,type=int,help="episode length in each epoch")
parser.add_argument("--save_model_freq",default=50,help="save model frequency (in num epoches)")
parser.add_argument("--lr",default=1e-3,type=float,help="learning rate for network")
parser.add_argument("--random_seed",default=0,type=float,help="set random seed if required (0 = no random seed)")
parser.add_argument("--eval_freq",default=100,type=int,help="evaluation frequency (in num epoches)")

parser.add_argument("--trainer", default="lstm",type=str, help="mlp, rnn, cnn, transformer")
parser.add_argument("--data_source",default="train",type=str,help="train, 2021, 2020")
parser.add_argument("--iso",default=argparse.SUPPRESS,type=str,help="PJM, CAISO, NYISO, MISO, ISONE")
# multi select choice: energy regulation reserve
parser.add_argument('--product',choices=['energy', 'regulation', 'reserve'],default=['energy', 'regulation', 'reserve'],nargs='+',help='Choose the market')
parser.add_argument("--soc",default=4,type=float,help="The fixed soc hour of the energy storage")
parser.add_argument("--degradation_cost",default=10.0,type=float,help="Battery degradation cost ($/MWH-cycle)")
parser.add_argument('--node',choices=['AECO'],default=['AECO'],nargs='+',help='Choose the node')

# [Modified] Add Training Mode for Ablation Study
parser.add_argument("--mode", choices=['default', 'da_only', 'rt_only'], default='default', 
                    help="Training mode: default (Dual Market), da_only (Day-Ahead Only), rt_only (Real-Time Only)")

parser.add_argument("--checkpoint",default=None,type=str,help="checkpoint path")


args = parser.parse_args()

# [Modified] Infer num_markets from product
args.num_markets = 0
if 'energy' in args.product: args.num_markets += 1
if 'regulation' in args.product: args.num_markets += 2
if 'reserve' in args.product: args.num_markets += 6

################################### Training ###################################
print("============================================================================================")
device = torch.device("cuda:"+str(args.cuda))
print("Device set to : " + str(torch.cuda.get_device_name(device)))
print("============================================================================================")
################################## set device ##################################


if __name__ == "__main__":
    print("============================================================================================")
    print("training environment name : " + args.env_name)


    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    exp_time_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir =  os.getcwd()+"/logs/" +args.env_name + '/'+ args.exp_name
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    log_dir = log_dir + '/' + exp_time_str + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### create new log file for each run
    log_f_name = log_dir + "log.csv"

    print("logging at : " + log_f_name)

    #### Save all hyperparameters to json file
    ### store args to param.json
    param_f_name = log_dir + "param.json"
    with open(param_f_name, 'w') as f:
        json.dump(vars(args), f)

   ################### checkpointing ###################

    directory = log_dir
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "{}.pth".format(0)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")

    print("num of parrallel environments(batch_size)",args.batch_size)
    print("total epoches trained: ",args.total_epoches)
    print("episode length: ",args.eps_len)
    print("model saved every ",args.save_model_freq, "epoches")
    print("optimizer learning rate : ", args.lr)
    print("Training Mode : ", args.mode)
    if args.random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    print("============================================================================================")

    ################# training procedure ################


    # initialize a PPO agent
    trainer_dict = {
        "lstm": LSTMTrainer
    }

    trainer = trainer_dict[args.trainer](batch_size=args.batch_size,seq_len = args.eps_len,learning_rate=args.lr,device=device,env_config=args.__dict__)
    
    if args.checkpoint:
        trainer.load(args.checkpoint)


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    
    # Create plot directory
    plot_dir = directory + "plots/"
    os.makedirs(plot_dir, exist_ok=True)

    # logging wandb
    wandb.init(
        project="meta-bidding",
        name=args.exp_name,
        config=vars(args)
    )
    wandb.run.log_code(".")


    for step in range(args.total_epoches):
        eps_rew = trainer.train_eps() 

        wandb.log({
            "train_reward": np.clip(eps_rew,-10,np.inf),
            "batch_soc_violation": trainer.env.current_soc_violation_freq
        }, step=step)
        print("Episode : {} \t\t Timestep : {} \t\t Train Reward : {}".format(step, step*args.batch_size*args.eps_len, eps_rew))
        log_f.write('{},{},{}\n'.format(step, step*args.batch_size*args.eps_len, eps_rew))
        log_f.flush()

        if step % args.eval_freq == 0:
            with torch.no_grad():
                eval_results = trainer.evaluate_eps()
            
            # Extract mean profit for logging
            mean_profit = eval_results['mean_profit']
            wandb.log({"eval_reward": np.clip(mean_profit,-10,np.inf)}, step=step)
            
            # Log per-market reward
            if 'reward' in eval_results and eval_results['reward'].ndim > 2:
                 avg_rew_per_mkt = np.mean(eval_results['reward'], axis=(0, 1))
                 log_dict = {f"eval/reward_mkt_{i}": avg_rew_per_mkt[i] for i in range(9)}
                 wandb.log(log_dict, step=step)
            
            print("Eval Reward : {}".format(mean_profit))
            
            # --- Plotting Logic ---
            try:
                # Data Slicing: Agent 0, Day 0 (First 288 steps)
                rt_action = eval_results['rt_action'][:288, 0, :] # (288, 9)
                soc = eval_results['soc'][:288, 0] # (288,)
                lmp = eval_results['lmp'][:288, 0, :] # (288, 9)
                lmp_da = eval_results['lmp_da'][:288, 0, :] # (288, 9)
                
                fig, axes = plt.subplots(5, 1, figsize=(14, 22), gridspec_kw={'height_ratios': [3, 3, 3, 3, 1.5]})
                
                # Subplot 1: Actions (Energy Market)
                axes[0].plot(rt_action[:, 0], label='RT Energy Action', color='blue')
                
                if 'da_action' in eval_results and eval_results['da_action'] is not None:
                    da_action = eval_results['da_action'][0, 0, :, :] # (9, 24)
                    # Repeat DA action to match 5-min resolution (24 -> 288)
                    da_action_energy = np.repeat(da_action[0, :], 12)
                    axes[0].plot(da_action_energy, label='DA Energy Plan', color='red', linestyle='--', alpha=0.7)
                
                axes[0].set_ylabel('Power (MW)')
                axes[0].set_title(f'Actions (Step {step})')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Subplot 2: SoC
                axes[1].plot(soc, label='SoC', color='green')
                axes[1].set_ylabel('SoC (0-1)')
                axes[1].set_ylim(-0.1, 1.1)
                axes[1].set_title('State of Charge')
                axes[1].grid(True, alpha=0.3)
                
                # Subplot 3: Price (Energy Market)
                axes[2].plot(lmp[:, 0], label='RT Price', color='orange')
                axes[2].plot(lmp_da[:, 0], label='DA Price', color='cyan', linestyle='--')
                axes[2].set_ylabel('Price ($/MWh)')
                axes[2].set_title('Energy Price')
                axes[2].set_xlabel('Time Step (5-min)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Prepare Revenue Data
                if 'rev_da' in eval_results and 'rev_rt_deviation' in eval_results:
                    # Sum over time for Agent 0
                    day_rev_da = np.sum(eval_results['rev_da'][:288, 0, :], axis=0) # (9,)
                    day_rev_rt_dev = np.sum(eval_results['rev_rt_deviation'][:288, 0, :], axis=0) # (9,)
                    
                    # Subplot 4: Energy Market Revenue (Stacked)
                    bar_width = 0.3
                    axes[3].bar(['Energy'], [day_rev_da[0]], width=bar_width, label='DA Revenue', color='crimson', alpha=0.6)
                    axes[3].bar(['Energy'], [day_rev_rt_dev[0]], width=bar_width, bottom=[day_rev_da[0]], label='RT Deviation', color='royalblue', alpha=0.6)
                    axes[3].set_title('Daily Revenue Breakdown (Energy Market)')
                    axes[3].set_ylabel('Revenue ($)')
                    axes[3].legend()
                    axes[3].grid(True, alpha=0.3, axis='y')
                
                # Subplot 5: Statistics
                axes[4].axis('off')
                stats_text = f"Mean Profit: {mean_profit:.2f}\n"
                axes[4].text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center')
                
                # Save Plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"eval_step_{step}_{timestamp}.png"
                save_path = os.path.join(plot_dir, filename)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close(fig)
                print(f"Plot saved to {save_path}")
                
            except Exception as e:
                print(f"Error plotting evaluation results: {e}")
                import traceback
                traceback.print_exc()

        if step % args.save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            checkpoint_path = directory + "{}.pth".format(step)
            print("saving model at : " + checkpoint_path)
            trainer.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

    # print total training time
    log_f.close()
    # env.close()
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    wandb.finish()