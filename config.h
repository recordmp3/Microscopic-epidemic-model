#ifndef CONFIG_H
#define CONFIG_H
#include<algorithm>
#include<map>
#include<cstring>
#include<string>
#include<vector>
#include<ctime>
#include<random>
#include<chrono>
#include<cmath>
#include<ctime>
#include<cstdlib>
using std::to_string;
using std::shuffle;
using std::rand;
using std::srand;
using std::default_random_engine;
using std::discrete_distribution;
using std::string;
using std::map;
using std::vector;
using std::max;
using std::min;

//global utils
bool hpn(double p);
extern std::random_device e;
extern std::uniform_real_distribution<double> u;
//global const mappings
extern map<int,string>m_rl;
extern map<string, int> m_hosp_policy;
extern map<string, int> m_obs;
extern map<int, string> m_obs2;
extern map<string, int> m_hs;
extern map<int, string> m_hs2;
extern map<string, int> m_al;
extern map<string, int> m_pt;
extern map<int, string> m_apt;
extern map<string, float>m_fac2;
extern map<float, string>m_fac;
extern map<string, int> m_ind_action;
extern map<string, int> m_require_activity_level;
extern map<string, int>m_shop;
extern vector<string>linear_l;
extern int l_hs_ill[8]; 
extern int l_obs_ill[4];
extern string l_must_only[2], l_local_only[1], l_limit_only[3], l_full_only[4];
extern string l_must[2], l_local[3], l_limit[6], l_full[10], l_facility[14], l_profitable[11], l_robotize[3], gov_fac_lst[13];
class Args{
    //int obs_size;
    public:
        map<string, int> action_size;
        int n_agent, action_repeat, stop_step;
        int hospitalized_capacity, hospital_testing_capacity, hosp_policy;
        
        float recreational_reward, community_reward;
        float work_reward;

        float gov_disinfect_cost, gov_test_cost, gov_robot_maintainance;
        float gov_death_penalty, gov_infection_penalty;
        float gov_reward_adjust_ratio, gov_infected_penalty_growth, gov_death_penalty_growth;
        int gov_action_size, gov_maximum_robot;
        float meta_death_penalty, meta_infection_penalty;
        int maximum_online_request;
        float mask_infected_rate, mask_infect_rate;
        int n_disinfected_last;
	    float disinfected_rate;
        vector<vector<int> > hospital_list; // the order of visiting each hospital.
        float p_inc2pre, p_ina2asy, p_pre2sym, p_hos, p_rec_sym;
        float p_sym2sev[3], p_sev2cri[3], p_cri2dea[3];
        int agent_count[4];
        float p_sev2rec, p_cri2rec, p_sev2cri_nhos[3], p_rec_asy;
        float p_deimm_asy, p_deimm_sym;
        float asy_pop_rate, asy_infect_rate, pre_infect_rate;
        float hpn;
        string name;	
        map<string, int> n_p, n_f;
        vector<int> type, age;
        map<string, vector<int> > graph;
        map<string, int> f_capacity_debug;
        map<string, vector<string> > name_f;
        map<string, vector<map<string, float> > > info_f; 
        float beta;
        Args();
        void init();
	float alpha;
};
//args_ind and args_gov does not appear here; they are at the python level.
#endif
