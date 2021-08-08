#include "config.h"
#include "individual.h"
#include "facility.h"
#include "city.h"
#include<map>
#include<vector>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<iostream>
#include<ctime>
#include<queue>
#include<utility>
using std::queue;
using std::pair;
#ifndef ENV_H
#define ENV_H
typedef vector<vector<float> > IndAction; // the collection of individual action.
typedef vector<float> GovAction; //Gov action
typedef map<string, map<string, float> > Obs; // this is the collection of all ind agent and gov"s observation.

class AllAction{
    public:
    IndAction ind;
    GovAction gov;
};

class Step_returns{
    public:
    vector<float> ind_obs; // ind, gov
    Obs gov_obs;
    vector<float> reward;
    vector<vector<int> > action_count;
    vector<int> done;
    vector<map<string, float> > info;
    map<string, vector<float> > graph;
};
class Env{
    public:
        double t1, t2, t3, t4;
        int n_agent;
        vector<Individual> ind;
        vector<int> new_ind_hs;
        map<string, vector<Facility>> fac;
        City city;
        queue<int> online_queue;
        int stop_step, s_n;
        Args args;
        map<string, int> n_p_hs;
        map<string, int> n_infected_new, n_infected;
        vector<int> gov_unchanged_days, gov_last_action;
        float new_death_average, new_infection_average;
        int last_new_deaths, last_new_infections;
        int gov_current_robot;
        int n_infected_all, overall_accurate_infected, overall_observed_infected;
        // FIXME: viewer not implemented
        vector<float> sum_r;
        map<string, vector<int> > graph;
        pair<vector<float>, Obs> reset(); // return type is the same as obs
        void init_epidemic(); // set initial infectants.
        int step_health_state(int id); // return new deaths
        Env();
        void init(Args &args);
        //observation is map<string, float>
        vector<float> deal_gov_action(vector<float> &action_gov);
        // return num_of_disinfected, num_of_tested, num_of_robots.
        bool shutdowned(string facility_type, int facility_id);
        int where_hospital(int ID, vector<float>& action);
        bool is_workable(int ID);
        bool is_outable(int ID);
        bool is_working(int ID, vector<float>& action);
        bool is_schooling(int ID, vector<float>& action);
        // is_dead is obsoleted.
        bool is_disinfected(string &facility_type, int facility_id);
        bool is_hospitalized(int ID);
        bool is_examined(int ID);
        void deal_general_facility(const string & facility_type, int facility_id);
        void deal_general(const string & facility_type, int facility_id);
        void deal_market(const string & facility_type, int facility_id);
        int activity_level(vector<float> &action);
        void deal_hospital(int facility_id);
        int deal_ind_action(IndAction &ind_action);// return newly_infected
        Step_returns step(AllAction actions);
        float parse_reward_ind(int id, vector<float> &action, int health_old);
        vector<float> parse_reward_gov(float changing_penalty,float disinfect_cost,int num_of_tested, int num_of_robots, int new_deaths, int new_infections);
        map<string, map<string, float> > parse_obs_gov();
        vector<float> parse_obs_ind(int id);
        map<string, vector<float> > get_graph_info();
	void gov_control_ind_and_fac();

        // FIXME: render, add_cell_geom and close are not implemented.
};
#endif
