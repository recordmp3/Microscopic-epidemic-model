from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.queue cimport queue
from libcpp.utility cimport pair
ctypedef vector[vector[float]] IndAction
ctypedef vector[float] GovAction
ctypedef map[string, map[string, float]] Obs

cdef extern from "facility.cpp":
    pass

cdef extern from "facility.h":
    cdef cppclass Facility:
        string type, name
        float longtitude, latitude
        int n_infected, n_infected_new
        int basic_capacity, adjusted_capacity, n_d, current
        float avg_frequency, infect_rate
        bool disinfected, shut, rotate
        vector[int] id_l
        int robots, unexamined
        Facility()
        Facility(map[string, float] info, string name)
        void reset()
        float efficiency()
        int operational_workers, full_workers
        float revenue, ind_reward; 

cdef extern from "city.cpp":
    pass

cdef extern from "city.h":
    cdef cppclass City:
        Args args
        map[string, int] n_p_hs
        map[string, int] n_infected_new, n_infected
        int n_infected_all
        void reset()
        City()

cdef extern from "individual.cpp":
    pass

cdef extern from "individual.h":
    cdef cppclass Individual:
        void init(int type)
        void reset()
        float R0
        int type, health, which_hospital, health_obs
        double p_not_infected
        double p_infect
        double infected_rate
        double supply_level
        bool get_infected_now, has_examined_ill
        bool examined, hospitalized
        vector[float] action
        map[string, int] has_gone
        int get_obs()
"""
cdef extern from "EnvCWrapper.cpp": # to includet these code.
    pass

cdef extern from "EnvCWrapper.h":
    cdef cppclass EnvCWrapper:
        EnvCWrapper() except +  # let C++ handle the exception.
        init(Args args) except +
        void* env
        Obs reset() except +
        Step_returns step_(AllAction action) except +
"""
cdef extern from "environment.cpp":
    pass

cdef extern from "environment.h":
    cdef cppclass AllAction:
        IndAction ind
        GovAction gov

    cdef cppclass Step_returns:
        Obs gov_obs
        vector[float] ind_obs
        vector[vector[int]] action_count
        vector[float] reward
        vector[int] done
        vector[map[string, float]] info
        map[string, vector[float]] graph

    cdef cppclass Env:
        double t1, t2, t3, t4
        int n_agent
        vector[Individual] ind
        vector[int] new_ind_hs
        map[string, vector[Facility]] fac
        City city
        queue[int] online_queue
        int stop_step, s_n
        Args args
        map[string, int] n_p_hs
        map[string, int] n_infected_new, n_infected
        vector[int] gov_unchanged_days, gov_last_action
        int last_new_deaths, last_new_infections
        int gov_current_robot
        int n_infected_all, action_repeat, overall_accurate_infected, overall_observed_infected
        vector[float] sum_r
        map[string, vector[vector[int]]] graph
        pair[vector[float], Obs] reset()
        void init_epidemic()
        int step_health_state(int id)
        void init(Args &args)
        Env()
        vector[float] deal_gov_action(vector[float] &action_gov)
        int where_recreational(int ID, vector[float] &action)
        bool shutdowned(string facility_type, int facility_id)
        int where_hospital(int ID, vector[float]& action)
        bool is_workable(int ID)
        bool is_outable(int ID)
        bool is_working(int ID, vector[float]& action)
        bool is_schooling(int ID, vector[float]& action)
        bool is_disinfected(string &facility_type, int facility_id)
        bool is_hospitalized(int ID)
        bool is_examined(int ID)
        void deal_debug(const string &facility_type, int facility_id)
        void deal_market(const string &facility_type, int facility_id)
        void deal_general(const string &facility_type, int facility_id)
        int activity_level(vector[float] &action)
        void deal_hospital(int facility_id)
        int deal_ind_action(IndAction &ind_action)
        Step_returns step(AllAction actions)
        float parse_reward_ind(int id, vector[float] &action, int health_old)
        vector[float] parse_reward_gov(float changing_penalty,float disinfect_cost,int num_of_tested, int num_of_robots, int new_deaths, int new_infections)
        Obs parse_obs_gov()
        vector[float] parse_obs_ind(int id)

cdef extern from "config.cpp":
    pass

cdef extern from "config.h":
    cdef cppclass Args:
        map[string, int] action_size;
        int n_agent, action_repeat, stop_step;
        int hospitalized_capacity, hospital_testing_capacity, hosp_policy;
        float hpn; 
        float recreational_reward, community_reward;
        float work_reward;

        float gov_disinfect_cost, gov_test_cost, gov_robot_maintainance;
        float gov_death_penalty, gov_infection_penalty;
        float gov_reward_adjust_ratio, gov_infected_penalty_growth, gov_death_penalty_growth;
        int gov_action_size, gov_maximum_robot;
        int maximum_online_request;

        float mask_infected_rate, mask_infect_rate;
        int n_disinfected_last;
        float disinfected_rate;
        
        float p_inc2pre, p_ina2asy, p_pre2sym, p_hos, p_rec_sym;
        float p_sym2sev[3];
        float p_sev2cri[3];
        float p_cri2dea[3];
        int agent_count[4];
        float p_sev2rec, p_cri2rec, p_sev2cri_nhos, p_rec_asy;
        float p_deimm_asy, p_deimm_sym;
        float asy_pop_rate, asy_infect_rate, pre_infect_rate;
        
        map[string, int] n_p, n_f;
        vector[int] type;
        map[string, vector[vector[int]]] graph;
        map[string, int] f_capacity_debug;
        map[string, vector[string]] name_f;
        map[string, vector[map[string, float]]] info_f; 

        float beta;
        Args();
        void init();
