import numpy as np
import time
import os
import torch
from datetime import datetime
time_str = str(datetime.now()).replace(':','-').replace(' ','--')
time_str = time_str[0:time_str.find(".")]

from env2 import * # importing cython.
from utils import onehot
from config import *
from a2c_ppo_acktr.envs import make_vec_envs

from DQN import DQN, Net
from Storage import RolloutStorage

def get_best_gpu(force = None):
    if force is not None:return force
    s = os.popen("nvidia-smi --query-gpu=memory.free --format=csv")
    a = []
    ss = s.read().replace('MiB','').replace('memory.free','').split('\n')
    s.close()
    for i in range(1, len(ss) - 1):
        a.append(int(ss[i]))
    print(a)
    best = int(np.argmax(a))
    print('the best GPU is ',best,' with free memories of ',ss[best + 1])
    return best

gpu_id = get_best_gpu()
print(gpu_id, 'gpu_id')
torch.cuda.set_device(gpu_id)
agent_range_pt = []

time_str = str(datetime.now()).replace(':', '-').replace(' ', '--')
time_str = time_str[0:time_str.find(".")]
his_r, his_epoch, his_vl, his_ent, his_gn, his_action = [], [], [], [], [], [[[], []], [[], []], [[], [], [], []], [[], [], []]] # reward history
his_activity_level = [[], [], [], []]
if __name__ == "__main__":
    save_path = os.path.join('results', time_str)
    file_path = os.path.join(save_path, 'files')
    os.makedirs(file_path)
    s = os.popen("cp -r `ls |grep -v results|grep -v graph_sorted.txt|xargs` " + file_path)
    g = open("T.txt", "w") 
    torch.set_num_threads(8)
    device = torch.device("cuda")
    
    action_space_chd = [2, 2]
    action_space_sch = [2, 2, 4, 3]
    action_space_wk = [2, 2, 4, 3]
    action_space_rtr = [2, 2, 4, 3]

    args_ind = Args_ind()
    env = env2(**{'time_str':time_str,  'save_path': save_path, 'rank': -1})
    obs_size = env.obs_size
    n_epi_per_train = args_ind.n_epi_per_train
    n_type = 4
    S_N_chd = Net(obs_size, action_space_chd)
    T_N_chd = Net(obs_size, action_space_chd)
    S_N_sch = Net(obs_size, action_space_sch)
    T_N_sch = Net(obs_size, action_space_sch)
    S_N_wk  = Net(obs_size, action_space_wk )
    T_N_wk  = Net(obs_size, action_space_wk )
    S_N_rtr = Net(obs_size, action_space_rtr)
    T_N_rtr = Net(obs_size, action_space_rtr)
    ind = [
            DQN(S_N_chd, T_N_chd, args_ind.alpha, lr = 1e-2, device = device),
            DQN(S_N_sch, T_N_sch, args_ind.alpha, lr = 1e-2, device = device),
            DQN(S_N_wk , T_N_wk , args_ind.alpha, lr = 1e-2, device = device),
            DQN(S_N_rtr, T_N_rtr, args_ind.alpha, lr = 1e-2, device = device)
          ]
    rollouts_pt = [RolloutStorage(env.stop_step, env.ac[m_pt['chd']], obs_size, len(action_space_chd), None),
                   RolloutStorage(env.stop_step, env.ac[m_pt['sch']], obs_size, len(action_space_sch), None),
                   RolloutStorage(env.stop_step, env.ac[m_pt['wk']], obs_size, len(action_space_wk), None),
                   RolloutStorage(env.stop_step, env.ac[m_pt['rtr']], obs_size, len(action_space_rtr), None)]
    rollouts_pt_all = [RolloutStorage(env.stop_step, env.ac[m_pt['chd']], obs_size, len(action_space_chd), None),
                   RolloutStorage(env.stop_step, env.ac[m_pt['sch']], obs_size, len(action_space_sch), None),
                   RolloutStorage(env.stop_step, env.ac[m_pt['wk']], obs_size, len(action_space_wk), None),
                   RolloutStorage(env.stop_step, env.ac[m_pt['rtr']], obs_size, len(action_space_rtr), None)]
    epoch = -1
    epo_action = [[0, 0], [0, 0], [0, 0, 0, 0], [0, 0, 0]]
    while 1:
        epoch += 1
        print("epoch", epoch)
        epo_r = []
        epo_tde = []
        if len(agent_range_pt) == 0:            
            s = 0
            for j in range(n_type): 
                agent_range_pt.append((s, s + env.ac[j]))
                s += env.ac[j]
        print(agent_range_pt)
        obs = env.reset()[0]
        for i in range(n_type):
            sta = agent_range_pt[i][0]
            en  = agent_range_pt[i][1]
            rollouts_pt[i].obs[0].copy_(torch.Tensor(obs[sta*obs_size:en*obs_size]).to(device).reshape(-1, obs_size))

        value = [[] for i in range(n_type)]
        action = [[] for i in range(n_type)]
        action_log_prob = [[] for i in range(n_type)]
        recurrent_hidden_states = [[] for i in range(n_type)]
        for i in range(env.stop_step * n_epi_per_train):
            actions = []
            for j in range(n_type): # input must be in accordance with "chd, sch, wk, med, rtr"
                with torch.no_grad():
                    action[j], value[j] = ind[j].act(rollouts_pt[j].obs[i].to(device))
                    t_action = action[j].reshape(-1).float().cpu()
                    actions.append(t_action)
            
            gov_action = [4., 2., 4., 4., 3., 3., 4., 0, 0, 0, 0, 0, 0, 0]
            # gov_action is deprecated and overridden by C++ code.
            actions.append(torch.Tensor(gov_action))
            actions = torch.cat(actions, -1)
            obs, _, reward, done, info, graph_info = env.step(actions)
            
            draw.generate_graph(graph_info, save_path, i)
            
            reward = torch.Tensor(reward)#.to(device)
            done = torch.Tensor(done)#.to(device)
            epo_r.append(reward[:env.n_agent].mean().cpu().numpy() * args_ind.gamma ** (i % env.stop_step))

            for j in range(n_type):
                sta = agent_range_pt[j][0]
                en = agent_range_pt[j][1]

                rollouts_pt[j].insert(torch.Tensor(obs[sta*obs_size:en*obs_size]).reshape(-1, obs_size),#.to(device), 
                    action[j],#to(device), 
                    value[j], 
                    reward[sta:en].reshape(-1, 1),
                    (1 - done[sta:en]).reshape(-1, 1),
                    (1 - done[sta:en]).reshape(-1, 1))
            if i % env.stop_step != env.stop_step - 1:
                for k in range(len(epo_action)):
                    for j in range(len(epo_action[k])):
                        epo_action[k][j] += env.epo_action[k][j]
            print(epo_action)
            if i % env.stop_step == env.stop_step - 1:
                for k in range(len(epo_action)):
                    for j in range(len(epo_action[k])):
                        his_action[k][j].append(epo_action[k][j] / (env.stop_step * env.n_agent))
                epo_action = [[0, 0], [0, 0], [0, 0, 0, 0], [0, 0, 0]]
        for i in range(n_type):
            with torch.no_grad():
                next_value = ind[i].get_value(rollouts_pt[i].obs[-1].to(device)).detach().cpu()
                rollouts_pt[i].compute_returns(next_value, args_ind.gamma, args_ind.lamda)
        # update permanent buffer & policy
        for i in range(n_type):
            ocu = args_ind.n_epi_ocu
            if epoch == 0:ocu = 1
            rollouts_pt_all[i].copy_from(rollouts_pt[i], 0, ocu)
            ind[i].learn(rollouts_pt_all[i], 100, epoch)
            rollouts_pt[i].after_update()
        
        torch.save(ind, os.path.join(save_path, "ind.pt"))

        his_epoch.append(epoch)
        his_r.append(np.mean(epo_r)) 
        draw.draw(his_epoch, [his_r], 'n_epoch', 'reward', "_reward", save_path, label=['ppo_reward'])

        np.save(os.path.join(save_path, "his_r.npy"), np.array([his_r]))
        draw.draw([i for i in range(n_epi_per_train*(epoch+1))], his_action[m_ind_action["protection"]], 'n_epoch', 'protection', "_frequency_pro", save_path, label = ['none', 'mask'])
        draw.draw([i for i in range(n_epi_per_train*(epoch+1))],his_action[m_ind_action["activity"]], 'n_epoch', 'activity', "_frequency_act", save_path, label = ['must', 'local', 'limit', 'full'])
        draw.draw([i for i in range(n_epi_per_train*(epoch+1))], [his_action[m_ind_action["hospital"]][1]], 'n_epoch', 'hospital', "_frequency_hos", save_path, label = ['go to hospital'])
        draw.draw([i for i in range(n_epi_per_train*(epoch+1))], his_action[m_ind_action["market"]], 'n_epoch', 'hospital', "_frequency_shop", save_path, label = ['none', 'offline', 'online'])
