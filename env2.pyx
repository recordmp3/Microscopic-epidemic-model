# distutils: language = c++
from EnvCWrapper cimport *
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
from utils import onehot
import numpy as np
import draw
import os
import gym
import gym.spaces
import time

gov_fac_lst = ["workplace", "school", "stadium", "cinema", "theatre", "library", "museum", "gym", "retail", "community", "restaurant", "supermarket"]
l_facility = ["workplace", "school", "hospital", "household", "stadium", "cinema", "theatre", "library", "museum", "gym", "retail", "community", "restaurant", "supermarket"]
m_ind_action = {"protection":0,"hospital":1,"activity":2,"market":3}
m_pt = {"chd":0,"sch":1,"wk":2,"rtr":3,"med":4}

cdef class env2():
    cdef dict __dict__
    cdef Env c_env

    def __cinit__(self, time_str, save_path, rank):
        
        cdef Args cargs # automatically generate all values in constructor.
        t = time.time()
        cargs.init()
        # No need for value transportations! all configs are in config.cpp now.
        self.s_n = 0
        self.rank = rank
        
        self.save_path = save_path
        self.c_env = Env()
        self.c_env.init(cargs)
        
        self.graph_counter = 0
        self.time_str = time_str
        self.graph_info_reset()
        self.chd_count = cargs.agent_count[m_pt["chd"]]
        self.n_agent = cargs.n_agent
        self.adult_count = self.n_agent - self.chd_count
        self.name_g = ['healthy', 'dead', 'incubation', 'immune', 'ill']
        self.name_h = [['sus'], ['dea'], ['inc', 'ina'], ['imm_s', 'imm_a'],['sym', 'asy', 'msy', 'csy', 'ssy']]
        self.name_f = l_facility
        
        obs_size = 29
        self.action_size = [2] * self.chd_count + [4] * self.adult_count + [14]
        self.obs_size = obs_size
        self.ac = [cargs.agent_count[m_pt['chd']],cargs.agent_count[m_pt['sch']],cargs.agent_count[m_pt['wk']],cargs.agent_count[m_pt['rtr']]]

    def obs2obsgov(self, obs): # deprecated. 
        obs_l = [obs[b'city'][b'infected'], obs[b'city'][b'dead'], obs[b'city'][b'robots'], obs[b'city'][b'current_robot'], obs[b'city'][b'avg_new_infected'], obs[b'city'][b'avg_new_death']]
        for i in [b"gov_last_action", b"gov_unchanged_days"]:
            for j in [b"workplace", b"school", b"full", b"limit", b"local", b"must"]:
                obs_l.append(obs[i][j])
        for i in gov_fac_lst:
            for j in range(self.c_env.args.n_f[i.encode('gbk')]):
                lst = obs[(i+'_'+str(j)).encode('gbk')]
                obs_l += lst.values()
        return obs_l

    @property
    def n_agent(self):
        return self.c_env.n_agent

    @n_agent.setter
    def n_agent(self,x):
        self.c_env.n_agent = x
    @property 
    def stop_step(self):
        return self.c_env.stop_step
    @stop_step.setter
    def stop_step(self,x):
        self.c_env.stop_step = x
    

    def seed(self, x):#!
        pass

    def get_n_uninfected(self):
       return sum([1 if self.c_env.ind[i].health_obs == 0 else 0 for i in range(self.n_agent)])

    def get_graph_test(self, save_path):
        os.makedirs(save_path)
        step_l = [i for i in range(self.stop_step)]
        
        draw.draw(step_l, self.n_p_hs_l, 'step', 'population', "infection status", save_path, label=self.name_g)
        np.save(os.path.join(save_path, "inf_stat.npy"), np.array([self.n_p_hs_l]))
        draw.draw(step_l, self.n_infected_new_l, 'step', 'newly_infected', "new infection prop", save_path, label=self.name_f)
        np.save(os.path.join(save_path, "new_inf.npy"), np.array([self.n_infected_new_l]))
        draw.draw(step_l, self.gov_disinfect_l, 'step', 'disinfect', "disinfection breakdown" + str(self.graph_counter), save_path,
                          label=['workplace', 'school', 'full', 'limit',  'local', 'must']) 
        draw.draw(step_l, self.gov_capacity_l, 'step', 'capacity', "capacity breakdown" + str(self.graph_counter), save_path,
                          label=['workplace', 'school', 'full', 'limit',  'local', 'must']) 
        draw.draw(step_l, [self.gov_robot], 'step', 'robot', 'robot deployment'+str(self.graph_counter), save_path, label=['robot'])
        draw.draw(step_l, [self.gov_test], 'step', 'test', 'test'+str(self.graph_counter), save_path, label=['test'])
        draw.draw(step_l, self.epo_action_l[m_ind_action['protection']], 'step', 'protection', "frequency_pro " + str(self.graph_counter), save_path,
                          label=['none', 'mask'])
        draw.draw(step_l, self.epo_action_l[m_ind_action['activity']], 'step', 'activity_level', "frequency_act " + str(self.graph_counter), save_path,
                          label=['must', 'local', 'limit', 'full'])
        draw.draw(step_l, [self.epo_action_l[m_ind_action['hospital']][1]], 'step', 'hospitalized', "frequency_hos " + str(self.graph_counter), save_path, label=['hospitalized'])
        draw.draw(step_l, self.epo_action_l[m_ind_action['market']], 'step', 'market method', "frequency_mar " + str(self.graph_counter), save_path, label=['none', 'online', 'offline'])
        draw.draw(step_l, [self.changing_penalty, self.gov_reward], 'step', 'reward', 'gov_reward'+str(self.graph_counter), save_path, label =['changing_penalty', 'reward'])
        
        self.graph_counter += 1
        return self.epo_action, self.epo_action[1]

    def graph_info_reset(self):
        self.n_p_hs_l = [[] for _ in range(5)]
        self.n_infected_new_l = [[] for i in range(14)]
        self.epo_action_l = [[[], []], [[], []], [[], [], [], []], [[], [], []]]
        self.epo_action = [[0, 0], [0, 0], [0, 0, 0, 0], [0, 0, 0]]
        self.gov_capacity_l = [[] for i in range(6)]
        self.gov_disinfect_l = [[]for i in range(6)]
        self.gov_robot, self.gov_test = [], []
        self.changing_penalty, self.gov_reward = [], []

    def collect_graph_info_per_step(self, action_count, gov_action):
        action = []
        tim = time.time()
        t = 0
        # infection status
        for j in range(len(self.name_h)):
            s = 0
            for k in range(len(self.name_h[j])):
                s += self.c_env.city.n_p_hs[self.name_h[j][k].encode("GBK")]
            self.n_p_hs_l[j].append(s / self.n_agent)
        # infected_new
        for j in range(len(self.name_f)):
            self.n_infected_new_l[j].append(self.c_env.city.n_infected_new[self.name_f[j].encode("GBK")] / self.n_agent)
        # action status
        for j in range(len(self.epo_action_l)):
            for k in range(len(self.epo_action_l[j])):
                self.epo_action_l[j][k].append(action_count[j][k] / self.n_agent)
                self.epo_action[j][k] = action_count[j][k]
        self.gov_capacity_l[0].append(gov_action[0] * 0.25)
        self.gov_capacity_l[1].append(gov_action[1] * 0.5)
        self.gov_capacity_l[2].append(gov_action[2] * 0.25)
        self.gov_capacity_l[3].append(gov_action[3] * 0.25)
        self.gov_capacity_l[4].append(gov_action[4] * 0.25 + 0.25)
        self.gov_capacity_l[5].append(gov_action[5] * 0.25 + 0.25)
        s = 0 if len(self.gov_robot) == 0 else self.gov_robot[-1]
        self.gov_robot.append(max(s + gov_action[6] - 4, 0))
        g = 0
        assert gov_action[7] in [i for i in range(5)] 
        if gov_action[7] == 0: h = 0
        elif gov_action[7] == 1: h = 0.001
        elif gov_action[7] == 2: h = 0.01
        elif gov_action[7] == 3: h = 0.05
        elif gov_action[7] == 4: h = 0.1  
        self.gov_test.append(g)
        for i in range(8, 14):self.gov_disinfect_l[i-8].append(gov_action[i])
         
    def step(self, action):
      
        action = action.tolist()
        self.s_n += 1
        cdef AllAction a
        # transportations between python and C++ actions
        a.ind.resize(self.n_agent)
        a.gov.resize(self.action_size[-1]) # government action size
        t = 0
        
        for i in range(self.n_agent): # the last two is for gov & meta
            a.ind[i].resize(self.action_size[i])
            for j in range(self.action_size[i]):
                a.ind[i][j] = action[t]
                t += 1
        for i in range(self.action_size[-1]):
            a.gov[i] = action[t]
            t += 1
        cdef Step_returns sr
        sr = self.c_env.step(a)
        # transportations between return and python format
        reward, done, info = sr.reward, sr.done, sr.info  # vector automatically transfrom into list.
        action_count = sr.action_count
        graph_info = sr.graph
        # drawing graphs
        self.collect_graph_info_per_step(action_count, a.gov)
        # info has  ind_
        info = {'changing_penalty':sr.info[self.n_agent][b'adjusted_changing_penalty']}
        self.changing_penalty.append(sr.info[self.n_agent][b'adjusted_changing_penalty'])
        self.gov_reward.append(sr.reward[-1])
        if self.s_n % self.stop_step == 0:info[b'graph'] = self.get_graph_test(self.save_path+'/'+str(self.s_n // self.stop_step)+str('_')+str(self.rank))
        gov_obs = self.obs2obsgov(sr.gov_obs)
        return sr.ind_obs, gov_obs, reward, done, info, graph_info
        
    def reset(self):
        cdef pair[vector[float], Obs] obs
        self.graph_info_reset()
        obs = self.c_env.reset()
        return obs.first, self.obs2obsgov(obs.second)

    def close(self):
        pass
    
