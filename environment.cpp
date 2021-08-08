#include"environment.h"
#include"individual.h"
#include<sys/time.h>
#include<utility>
#include<queue>
#include<thread>
#include<mutex>
#include<cmath>
#include<fstream>
#define forr(i,n) for(int i = 0;i < n; i ++) // forr = for range
#define forr2(i,s,t) for(int i = s;i < t; i ++)
#define forr3(i,s,t,d) for(int i = s;i < t; i += d)
using std::queue;
using std::cout;
using std::endl;
using std::thread;
using std::mutex;
using std::ofstream;
using std::ios;
int obs_size = 29; // observation size 
/*******args of configs********/

/*
bool is_calibration; // if is true, then the reward would be 
float r_ill; // R_{ill}
bool SC; // whether strong control 
float p_discover; //prob if finding a sym individual
float p_contact_tracing; //efficiency of contact tracing
bool information_disclosure; // whether ID is executed
*/

    // args of QT

        // strong QT

        
        bool is_calibration = false;
        float r_ill = 1e3; // 3e3, 1e4, 2.5e4
        bool SC = false;
        float p_discover = 1;
        float p_contact_tracing = 0.4;
        bool information_disclosure = true;
        

        // weak QT

        /*
        bool is_calibration = false;
        float r_ill = 1e3; // 3e3, 1e4, 2.5e4
        bool SC = false;
        float p_discover = 1.0 / 3;
        float p_contact_tracing = 0.;
        bool information_disclosure = true;
        */
        
        // no control

        /*
        bool is_calibration = false;
        float r_ill = 1e3; // 3e3, 1e4, 2.5e4
        bool SC = false;
        float p_discover = 0.0;
        float p_contact_tracing = 0.;
        bool information_disclosure = true;
        */

    // args of ID

        // ID

        /*
        bool is_calibration = false;
        float r_ill = 1e3; // 3e3, 1e4, 2.5e4
        bool SC = false;
        float p_discover = 0.;
        float p_contact_tracing = 0.;
        bool information_disclosure = true;
        */

        // no control

        /*
        bool is_calibration = false;
        float r_ill = 1e3; // 3e3, 1e4, 2.5e4
        bool SC = false;
        float p_discover = 0.;
        float p_contact_tracing = 0.;
        bool information_disclosure = false;
        */
    
    // args of calibration

        /*
        bool is_calibration = true;
        float r_ill = 0; # whatever a value
        bool SC = true;
        float p_discover = 0.;
        float p_contact_tracing = 0.;
        bool information_disclosure = false;
        */

/*******global variables********/


int ca_su = 0;
int cl_su = 0;
int qu_su = 0;
int D_day = -1; // D_day = 0 means days after first infection discovered.
double p_mar = 0;
int phase = 0;
double Rill = 0;
mutex m[1200010];
queue<thread> q_tr;
int first_episode = 1;
float asy_n = 0, pre_n = 0;
int tot_death_breakdown[3] = {0, 0, 0}, tot_hosp_breakdown[3] = {0, 0, 0};
int today_avg_age = 0, tot_chd = 0, tot_wk = 0, tot_rtr = 0, tot_off_hosp_death = 0, daily_clip_ill_reward = 0;
int daily_new_cases = 0, daily_new_hosp = 0, tot_new_hosp = 0, tot_new_cases = 0, daily_recovery = 0, tot_deaths = 0, tot_recovery = 0, tot_severe = 0, tot_kick = 0;
mutex BGL; //big global lock for all global statistic numbers.
string file_name = "b500rill1000_alpha1_3_ocu1_3_lr1e-2to1e-3_m01_s440_ct04_1_New7";
ofstream fout((string("simu_") + file_name + string(".txt")).c_str(), ios::out);
ofstream fout2((string("debug_simu_") + file_name + string(".txt")).c_str(), ios::out);

/*******utils functions********/

long get_time() {
    struct timeval timeStart;
    gettimeofday(&timeStart, NULL );
    return timeStart.tv_usec;
}
int agenum(int x){
    if(x == m_pt["chd"] || x == m_pt["sch"]) return 0;
    else if(x == m_pt["wk"] || x == m_pt["med"]) return 1;
    return 2;
}
template<typename T> vector<T> list(T a, T b) {
    vector<T> x;
    x.push_back(a);
    x.push_back(b);
    return x;
}

template<typename T> vector<T> list(T a, T b, T c) {
    vector<T> x;
    x.push_back(a);
    x.push_back(b);
    x.push_back(c);
    return x;
}

template<typename T> vector<T> list(T a, T b, T c, T d) {
    vector<T> x;
    x.push_back(a);
    x.push_back(b);
    x.push_back(c);
    x.push_back(d);
    return x;
}

template<typename T> vector<T> list(T a, T b, T c, T d, T e) {
    vector<T> x;
    x.push_back(a);
    x.push_back(b);
    x.push_back(c);
    x.push_back(d);
    x.push_back(e);
    return x;
}

template<typename T> vector<T> list(T a, T b, T c, T d, T e, T f) {
    vector<T> x;
    x.push_back(a);
    x.push_back(b);
    x.push_back(c);
    x.push_back(d);
    x.push_back(e);
    x.push_back(f);
    return x;
}
template<typename T> vector<T> list(T a, T b, T c, T d, T e, T f, T g, T h) {
    vector<T> x;
    x.push_back(a);
    x.push_back(b);
    x.push_back(c);
    x.push_back(d);
    x.push_back(e);
    x.push_back(f);
    x.push_back(g);
    x.push_back(h);
    return x;
}

template<typename T> bool in(T a, vector<T> b){
    for (auto x : b)
    if (a == x)return true;
    return false;
}

template<typename T> bool in(T a, T* b, int n){
    forr (i, n)
    if (a == b[i])return true;
    return false;
}

template<typename T, typename T2> void set(vector<T> &a, T2 b){
    forr(i, a.size())
    a[i] = b;
}

vector<float> parse_obs_ind_(Env* this_, int id);

template<class T> void out(const vector<T> &a){
    forr(i, a.size())cout<<a[i]<<' ';cout<<endl;
}

void mt_wait(){
    while (!q_tr.empty()){
        q_tr.front().join();
        q_tr.pop();
    }
}

/*******constructor of Env, modules of Env are built here ********/

Env::Env(){}
void Env::init(Args &args){
    int rnd_sd = get_time();
    cout<<"set random_seed "<<rnd_sd<<endl;
    srand(rnd_sd);
    this->t1 = this->t2 = this->t3 = this->t4 = 0;
    this -> n_agent = args.n_agent;
    cout<<"Env init:"<<args.n_agent<<endl;
    this -> graph = args.graph;
    forr(i, args.n_agent){
        this -> ind.push_back(Individual());
        this -> ind[i].init(args.type[i], args.age[i]); // individual state
    }
    this -> new_ind_hs = vector<int>(args.n_agent, 0);
    this -> fac = map<string, vector<Facility> >();
    this -> city = City();
    this -> stop_step = args.stop_step;
    this -> s_n = 0;
    this -> last_new_infections = 0, this -> last_new_deaths = 0, this -> new_infection_average = 0, this -> new_death_average = 0; 
    this -> gov_last_action.clear();
    this -> gov_unchanged_days.clear();
    this -> gov_current_robot = args.gov_maximum_robot;
    while(!this->online_queue.empty())online_queue.pop();

    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
        const string i = it->first;
        this -> fac[i] = vector<Facility>();
        forr(j, args.n_f[i]){
            this -> fac[i].push_back(Facility(args.info_f[i][j], args.name_f[i][j]));  // differentiate facilities
        }
    }
    vector<string> l_worker = {"hospital", "workplace"};
    forr(i, args.n_agent)
        for (string j : l_worker){
            if (args.graph[j][i] != -1){
                this -> fac[j][args.graph[j][i]].full_workers += 1;
                this -> fac[j][args.graph[j][i]].operational_workers += 1;
            }
        }
    forr(i, args.n_f["hospital"])
        this -> fac["hospital"][i].current = 0;
    this -> args = args;
    this -> sum_r = vector<float>(args.n_agent + 1, 0);
    this -> overall_observed_infected = 0, this -> overall_accurate_infected = 0;
    return;
}

/*******reset function of a gym environment (executed at the beginning of each episode), modules of Env are reset here ********/

pair<vector<float>, Obs> Env::reset(){
    Args &args = this -> args;
    D_day = -1;
    forr(i, args.n_agent){
        this -> ind[i].init(args.type[i], args.age[i]); // individual state
    }
    tot_new_cases = 0, tot_new_hosp = 0, tot_recovery = 0, tot_severe = 0, tot_off_hosp_death = 0;
    phase = 0;
    tot_deaths = 0, tot_kick = 0;
    tot_chd = 0, tot_wk = 0, tot_rtr = 0, today_avg_age = 0;
    tot_death_breakdown[0] = tot_death_breakdown[1] = tot_death_breakdown[2] = 0;
    tot_hosp_breakdown[0] = tot_hosp_breakdown[1] = tot_hosp_breakdown[2] = 0;
    this -> new_ind_hs = vector<int>(args.n_agent, 0);
    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
        string i = it->first;
        forr(j, args.n_f[i]){
		this -> fac[i][j].n_infected_d = vector<int>();
		this -> fac[i][j].n_infected_d2 = vector<int>();
		this -> fac[i][j].infected_id = vector<int>();
	}
    }
    this -> fac.clear();
    this -> city = City();
    this -> s_n = 0;
    this -> last_new_infections = 0, this -> last_new_deaths = 0, this -> new_infection_average = 0, this -> new_death_average = 0; 
    this -> gov_current_robot = args.gov_maximum_robot;
    this -> gov_last_action.clear();
    this -> gov_unchanged_days.clear();
    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
        string i = it->first;
        forr(j, args.n_f[i]){
            this -> fac[i].push_back(Facility(args.info_f[i][j], args.name_f[i][j]));  // differentiate facilities
        }
    }
    vector<string> l_worker = {"hospital", "workplace"};
    forr(i, args.n_agent)
        for (string j : l_worker)
            if (args.graph[j][i] != -1){
                this -> fac[j][args.graph[j][i]].full_workers += 1;
                this -> fac[j][args.graph[j][i]].operational_workers += 1;
            }
    
    forr(i, args.n_f["hospital"])
        this -> fac["hospital"][i].current = 0;
    
    set(sum_r, 0);
    this -> overall_observed_infected = 0, this -> overall_accurate_infected = 0;
    this -> init_epidemic();
    while(!this->online_queue.empty())online_queue.pop();
    vector<float> obs;
    forr(i, this->n_agent){
        vector<float> v = parse_obs_ind_(this, i);
        obs.insert(obs.end(), v.begin(), v.end());
    }
    Obs gov_obs = this -> parse_obs_gov();
    D_day = -1;
    return make_pair(obs, gov_obs);
}

/******* the epidemic is initialized here (which people will be in incubating and symptomatic states are initialized) ********/

void Env::init_epidemic(){
    int id_com = rand() % this -> args.n_f["community"];
    id_com = 28; // the 28th block group is near to the West Pan Hospital.
    forr(i, 2){ // 2 people sym
	int id = rand() % this -> args.n_agent;
	if (this -> graph["community"][id] != id_com || this->ind[id].health != m_hs["sus"]){
	    i -= 1;
	    continue;
	}
        this -> ind[id].health = m_hs["sym"];
        this -> ind[id].health_obs = this -> ind[id].get_obs();
    }
    forr(i, 8){ // 8 people inc
    	int id = rand() % this -> args.n_agent;
	if(this->graph["community"][id] != id_com || this->ind[id].health != m_hs["sus"]){i-=1;continue;}
	this->ind[id].health = m_hs["pre"];
        this->ind[id].health_obs = this -> ind[id].get_obs();
    }
}

/******* health state development is calculated here ********/

int Env::step_health_state(int id){
    int new_deaths = 0;
    if (this -> ind[id].health == m_hs["inc"]){
        if (hpn(this -> args.p_inc2pre)){
            if (hpn(this -> args.asy_pop_rate))
                this -> ind[id].health = m_hs["ina"];
            else
                this -> ind[id].health = m_hs["pre"];
        }
            
    }
    else if (this -> ind[id].health == m_hs["ina"]){
        if (hpn(this -> args.p_ina2asy)){
           if (hpn(0.5))
               this -> ind[id].health = m_hs["imm_a"];
           else
               this -> ind[id].health = m_hs["asy"];
	}
    }
    else if (this -> ind[id].health == m_hs["pre"]){
        if (hpn(this -> args.p_pre2sym))
           this -> ind[id].health = m_hs["sym"];
    }
    else if (this -> ind[id].health == m_hs["asy"]){
        if (hpn(this -> args.p_rec_asy)){
            this -> ind[id].health = m_hs["imm_a"];
            this -> ind[id].examined = false;
        }
    }
    else if (this -> ind[id].health == m_hs["sym"]){
        if (hpn(this -> args.p_hos)){
            float p = u(e);
            if (p <= this -> args.p_sym2sev[agenum(this -> ind[id].type)]){
                this -> ind[id].health = m_hs["ssy"];
		BGL.lock();
		tot_severe++;
		BGL.unlock();
	    }
            else
                this -> ind[id].health = m_hs["msy"];
        }
    }
    else if (this -> ind[id].health == m_hs["msy"]){  // mild symptom does not need to be treated at hospital; if indeed go to hospital, they will not be affected by efficiency.
        if (hpn(this -> args.p_rec_sym)){
            this -> ind[id].health = m_hs["imm_s"];
            this -> ind[id].has_examined_ill = false;
            if (this -> ind[id].hospitalized){
                int hosp = this -> ind[id].which_hospital;
                this -> fac["hospital"][hosp].current -= 1;
                this -> ind[id].hospitalized = false;
                this -> ind[id].which_hospital = -1;
            }
        }
    }

    else if (this -> ind[id].health == m_hs["ssy"]){  // severe symptom
        if ( ! this -> ind[id].hospitalized){  // if not hospitalized, then it will become critical
            if (hpn(this -> args.p_sev2cri_nhos[agenum(this->ind[id].type)])){
                this -> ind[id].health = m_hs["csy"];
		BGL.lock();
                tot_off_hosp_death++;
		BGL.unlock();
            }
        }
        else{
            int hosp = this -> ind[id].which_hospital;
            if (hpn(this -> args.p_sev2rec * this -> fac["hospital"][
                hosp].efficiency())){  // if hospitalized, then it will eventually recover
                if(hpn(this->args.p_sev2cri[agenum(this->ind[id].type)]))this->ind[id].health = m_hs["csy"];
                else{
                    this -> ind[id].health = m_hs["imm_s"];
                    this -> ind[id].has_examined_ill = false;
                    this -> fac["hospital"][hosp].current -= 1;
                    this -> ind[id].hospitalized = false;
                    this -> ind[id].which_hospital = -1;
                }
            }
        }
    }

    else if (this -> ind[id].health == m_hs["csy"]){  // critical symptom (deprecated)
        this -> ind[id].health = m_hs["dea"];
        new_deaths += 1;
        BGL.lock();
        tot_death_breakdown[agenum(this->ind[id].type)]++;
	BGL.unlock();
    }
    else if (this -> ind[id].health == m_hs["imm_a"]){
        if (hpn(this -> args.p_deimm_asy))
        this -> ind[id].health = m_hs["sus"];
    }
    else if (this -> ind[id].health == m_hs["imm_s"]){
        if (hpn(this -> args.p_deimm_sym))
        this -> ind[id].health = m_hs["sus"];
    }
    return new_deaths;
}

/******* government strategies are executed here ********/
/*  we implemented very sophisticated government strategies but didn't use them in our paper. 
    So, the function is now only used to decide which facilities will be closed, and only the global variable SC can change the function's effect, the other codes and returns has no meaning now.
    In the paper, there are 2 strategies related to this function: 
        doing nothing (SC = False -> all facilities are open)
        strong control (used in calibration settings, the same as reality: SC = True -> some facilitis are closed (details are in the appendix))
*/

vector<float> Env::deal_gov_action(vector<float> &action_gov){
    float a = action_gov[0] * 0.25; //wk full = 4
    float b = action_gov[1]; //sc 0 close, 1 open(optional), 2 open
    float c = action_gov[2] * 0.25; // full
    float d = action_gov[3] * 0.25; // limit
    float e = action_gov[4] * 0.25 + 0.25; // local
    float f = action_gov[5] * 0.25 + 0.25; // must cap
    action_gov[0] = a;
    action_gov[1] = b;
    action_gov[2] = c;
    action_gov[3] = d;
    action_gov[4] = e;
    action_gov[5] = f;
    float g = action_gov[6] - 4; // robot
    float h;
    if (action_gov[7] == 0) h = 0;
    else if (action_gov[7] == 1)h = 0.001;
    else if (action_gov[7] == 2)h = 0.01;
    else if (action_gov[7] == 3)h = 0.05;
    else if (action_gov[7] == 4)h = 0.1;

    float dwork = action_gov[8], dschool = action_gov[9], dfull = action_gov[10], dlim = action_gov[11], dloc = action_gov[12], dsup = action_gov[13]; // disinfected workplace/recreational/school/community
    // the above mapping is mostly just for readability except for g, h.

    float changing_penalty = 0, disinfect_cost = 0;
    int num_of_disinfected = 0;
    int num_of_tested = 0;
    int num_of_robots = 0, robot_for_disinfect = 0;
    
    if(this -> gov_unchanged_days.size() == 0){ //initialization
         this -> gov_unchanged_days.resize(this -> args.gov_action_size, 0);
         this -> gov_last_action.resize(this -> args.gov_action_size);
         forr(i, this -> args.gov_action_size){
             this -> gov_last_action[i] = action_gov[i];
         }
    }
    
    forr(i, this -> args.gov_action_size){
	if(action_gov[i] == this -> gov_last_action[i]) this->gov_unchanged_days[i]++;
        else{
        	if(i < 6) changing_penalty += 60.0 / float(this -> gov_unchanged_days[i]);
                this -> gov_unchanged_days[i] = 1;
                this -> gov_last_action[i] = action_gov[i];
        }
    }
    for(string i: l_robotize)
        forr(j, this -> args.n_f[i]) 
            num_of_robots += this -> fac[i][j].robots;
    //cout<<"previously deployed robots:"<<num_of_robots<<" current plan:"<<g<<endl;
    int ac = 0, dx = 0;
    vector<string> l_disinfection = {"workplace", "school"};
    vector<int>l_dx = {8, 9};
    vector<int> l_ac = {0, 1};
    for(string i : l_full){
        l_disinfection.push_back(i);
        l_dx.push_back(13 - m_require_activity_level[i]); // 3 -- 10 full, 2 -- 11 limit, 1 -- 12 local, 0 -- 13 must
        l_ac.push_back(5 - m_require_activity_level[i]); // 3 -- 2, 2 -- 3, 1 -- 4 ,0 -- 5
    }
    
    for(string i : l_disinfection){
    	forr(j, this -> args.n_f[i]){
            this->fac[i][j].disinfected = false;
            if((!SC) || D_day < 10) 
            {
                this->fac[i][j].shut=false;this->fac[i][j].adjusted_capacity = this->fac[i][j].basic_capacity;
            }
            else if(D_day < 62){
                if(i == "workplace")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.25);
                else if (i == "community")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.55);
                else if (i == "supermarket" || i == "retail")this->fac[i][j].adjusted_capacity = this->fac[i][j].basic_capacity;
                else this->fac[i][j].adjusted_capacity = 0;
                this->fac[i][j].shut = bool(this->fac[i][j].adjusted_capacity == 0);
            }
            else if(D_day < 81){
                if(i == "workplace")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.5);
                else if(i == "community")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.55);
                else if(i == "supermarket" || i == "retail")this->fac[i][j].adjusted_capacity = this->fac[i][j].basic_capacity;
                else if(i == "restaurant")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.25);
                else this->fac[i][j].adjusted_capacity = 0;
                this->fac[i][j].shut = bool(this->fac[i][j].adjusted_capacity == 0);
            }
            else if(D_day < 109){
                if(i == "workplace") this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.75);
                else if(i == "supermarket" || i == "retail") this->fac[i][j].adjusted_capacity = this->fac[i][j].basic_capacity;
                else if(i == "community")this->fac[i][j].adjusted_capacity = this->fac[i][j].basic_capacity;
                else if (i == "restaurant")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.25);
                else if (i == "school")this->fac[i][j].adjusted_capacity = 0;
                else this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.5);
                this->fac[i][j].shut = bool(this->fac[i][j].adjusted_capacity == 0);
            }
            else if(D_day < 116){
                if(i == "workplace") this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.75);
                else if(i == "supermarket" || i == "retail") this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity);
                else if(i == "community")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.55);
                else if(i == "restaurant")this->fac[i][j].adjusted_capacity = 0;
                else if(i == "school")this->fac[i][j].adjusted_capacity = 0;
                else this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.5);
                this->fac[i][j].shut=bool(this->fac[i][j].adjusted_capacity == 0);
            }
            else if(D_day < 123){
                if(i == "workplace") this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.75);
                else if(i == "supermarket" || i == "retail")this->fac[i][j].adjusted_capacity = this->fac[i][j].basic_capacity;
                else if(i == "community")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.55);
                else if(i == "restaurant")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.1);
                else if(i == "school")this->fac[i][j].adjusted_capacity = 0;
                else this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.5);
                this->fac[i][j].shut = bool(this->fac[i][j].adjusted_capacity == 0);
            }
            else{
                if(i == "workplace") this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.75);
                else if(i == "supermarket" || i == "retail") this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity);
                else if(i == "community")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.55);
                else if(i == "restaurant")this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.35);
                else if(i == "school")this->fac[i][j].adjusted_capacity = 0;
                else this->fac[i][j].adjusted_capacity = int(this->fac[i][j].basic_capacity * 0.5);
                this->fac[i][j].shut = bool(this->fac[i][j].adjusted_capacity == 0);
            }
	    }
    }
    vector<int> healthy = vector<int>();
    forr(i, this -> n_agent){
	if(!in(this -> ind[i].health_obs, l_obs_ill, sizeof(l_obs_ill)/sizeof(l_obs_ill[0])))healthy.push_back(i);
    }
    int l = healthy.size();
    std::random_shuffle(healthy.begin(), healthy.end());
    forr(i, int(l * h)) {this -> ind[healthy[i]].examined = true; if(!in(this->ind[healthy[i]].health, {m_hs["sus"], m_hs["imm_a"], m_hs["imm_s"]})) this -> ind[healthy[i]].has_examined_ill = true;}
    if (g > 0){  // deploy robots
        // first deploy to hospitals, then deploy to workplace
        g = min(g, float(this -> gov_current_robot));
        for (string key : {"hospital", "workplace"})
            forr(i, this -> args.n_f[key]){
                float s = min(this -> fac[key][i].operational_workers / 3, this -> fac[key][i].full_workers - this -> fac[key][i].operational_workers - this -> fac[key][i].robots);
                float v = min(g, s);
                num_of_robots = num_of_robots + v;
                this -> fac[key][i].robots = this -> fac[key][i].robots + v;
                g = g - v;
                this -> gov_current_robot -= v;
            }
    }        
    else if (g < 0){  // recall robots
        // first recall from workplace, then recall from hospital
        for (string key : {"workplace", "hospital"}) // order matters here!
            forr(i, this -> args.n_f[key]){
                float v = min(-g, (float)this -> fac[key][i].robots);
                num_of_robots = num_of_robots - v;
                this -> fac[key][i].robots = this -> fac[key][i].robots - v;
                g = g + v;
                this -> gov_current_robot += v;
            }
    }
    this -> gov_current_robot += robot_for_disinfect;
    
    return list((float)num_of_tested, (float)num_of_robots, changing_penalty, disinfect_cost); // to decide reward
}

bool Env::shutdowned(string facility_type, int facility_ID){
    return this -> fac[facility_type][facility_ID].shut;
}

/******* find which hospital the individual is in  (as a patient or medicare provider) ********/
int Env::where_hospital(int ID, vector<float> &action){
    if (this -> is_hospitalized(ID))
        return this -> ind[ID].which_hospital;
 
    if (this -> ind[ID].type == m_pt["med"])  // medicare worker
        return this -> graph["hospital"][ID];
    else if (action[m_ind_action["hospital"]] == 1){
        int N = this -> args.hospital_list[ID].size(), rec = -1;
        for(int i = N - 1; i >= 0; i--){  //choose a hospital from hospital list.
            int x = args.hospital_list[ID][i];
            if(this -> fac["hospital"][x].current < this -> fac["hospital"][x].hospitalized_capacity) rec = x;
        }
        if(rec == -1) rec = 0;
        return rec;
    }
    else return -1;
}

bool Env::is_workable(int ID){  // can go to work
    if (!this -> ind[ID].has_examined_ill)
        // the asymptomatic patient is not allowed to work.
        return in(this -> ind[ID].health_obs, list(m_obs["healthy"], m_obs["immune"]));
    else
        return in(this -> ind[ID].health_obs, list(m_obs["asymptomatic"], m_obs["mild"], m_obs["healthy"], m_obs["immune"]));  // FIXME
}

bool Env::is_outable(int ID){  // can go out

    return in(this -> ind[ID].health_obs, list(m_obs["asymptomatic"], m_obs["mild"], m_obs["healthy"], m_obs["immune"]));
}

bool Env::is_working(int ID, vector<float> &action){
    //# if you go to hospital then you cannot normally go to work today.
    if (this -> ind[ID].hospitalized || action[m_ind_action["hospital"]] || (! this -> is_workable(ID))) return false;
    if (this -> graph["workplace"][ID] == -1) return false;
    if ( ! this -> shutdowned("workplace", this -> graph["workplace"][ID]))
        return true;
    return false;
}

bool Env::is_schooling(int ID, vector<float> &action){
    //# if you go to hospital then you cannot normally go to school today.
    if (this -> ind[ID].hospitalized || action[m_ind_action["hospital"]] || ( ! this -> is_workable(ID)))
        return false;
    if (this -> graph["school"][ID] == -1)return false;
    if (this -> fac["school"][graph["school"][ID]].rotate && action[m_ind_action["activity"]] == 0) return false;
    if (this -> shutdowned("school", this -> graph["school"][ID]))
        return false;
    return true;
}

bool Env::is_disinfected(string &facility_type, int facility_ID){
    return this -> fac[facility_type][facility_ID].disinfected;
}

bool Env::is_hospitalized(int ID){
    return this -> ind[ID].hospitalized;
}

bool Env::is_examined( int ID){
    return this -> ind[ID].examined;
}

/******* calculate the probability of each individual being infected in each facility ********/

void Env::deal_general_facility(const string &facility_type, int facility_id){
    double p = 0;
    Facility *fac = &(this -> fac[facility_type][facility_id]);
    fac -> p = 0;
    fac -> p2 = 0;
    if (fac -> close){
	    return;
    }
    double a = fac -> infect_rate * this -> args.beta * ((1.0 * fac->id_l.size()/fac->basic_capacity) / fac->basic_capacity);
    if(facility_type != "supermarket" and facility_type != "retail") 
	    a *= fac -> avg_frequency;
    float disinfect_situation = (fac -> disinfected) ? this -> args.disinfected_rate : 1;
    double p_asy = 0, p_pre = 0;
    double p_all = 0;
    vector<int> ill_people = vector<int>();
    for(int x: fac -> id_l){
        float comm_situation = 1;
        if(facility_type == "community")comm_situation = this->ind[x].action[m_ind_action["activity"]] * 0.5;
	p_all += this -> ind[x].p_infect;
        p += a * this -> ind[x].p_infect * disinfect_situation * comm_situation;       
	if (this -> ind[x].health == m_hs["asy"])
        p_asy += a * this -> ind[x].p_infect * disinfect_situation * comm_situation; 
	if(this->ind[x].health == m_hs["pre"])
	p_pre += a * this -> ind[x].p_infect * disinfect_situation * comm_situation;
    
	if(this->ind[x].p_infect > 0)ill_people.push_back(x);
    }
    fac -> p = p;
    fac -> p2 = p - p_asy - p_pre;
    if(p == 0)return;
    fac -> unexamined = 0;
    int tot_case = 0;
    vector<int> inf_id;
    for(int x: fac -> id_l){
        double t = p;
	m[x].lock();
    if (this -> ind[x].health == m_hs["sus"]){
        t *= this -> ind[x].infected_rate;
        if (this -> ind[x].action[m_ind_action["protection"]] == 1) t *= this -> args.mask_infected_rate;
        if (t > 1) t = 1;
        
        this -> ind[x].p_not_infected *= (1 - t);
        if (facility_type == "retail" or facility_type == "supermarket")
            this -> ind[x].p_not_infected_fac[4] *= (1 - t);
        else
            this -> ind[x].p_not_infected_fac[m_require_activity_level[facility_type]] *= (1 - t);
        if (hpn(t)){
            inf_id.push_back(x);
            fac -> infected_id.push_back(x);
            asy_n += p_asy / p;
            pre_n += p_pre / p;
            BGL.lock();
            this -> new_ind_hs[x] = m_hs["inc"];
            fac -> n_infected_new += 1;
            daily_new_cases += 1;
            today_avg_age += this->ind[x].age;
            if(this->ind[x].age<18)tot_chd++;
            else if(this->ind[x].age<65)tot_wk++;
            else tot_rtr++;
            tot_case += 1;
            this -> city.n_infected_new[fac -> type] += 1;
            BGL.unlock();
        }
    }
	m[x].unlock();
    }
   if (fac -> type == "household") return;
   if (D_day >= 0)
   fac -> n_infected_d[D_day] += tot_case;
   std::random_shuffle(inf_id.begin(), inf_id.end());
   std::random_shuffle(ill_people.begin(), ill_people.end());
   int now = 0;
   double p_now = 0;
   if (inf_id.size() == 0)return;
   forr(i, ill_people.size()){
       int x = ill_people[i];
       p_now += this -> ind[x].p_infect / p_all;
       while((now + 1.0) / inf_id.size() < p_now){
	   this -> ind[x].infect_id.push_back(inf_id[now]);
	   now += 1;
       }
   }
}

/******* update supply_level state of each individual ********/    

void Env::deal_market(const string& facility_name, int facility_id){
    this -> deal_general_facility(facility_name, facility_id);
    int N = this -> fac[facility_name][facility_id].id_l.size();
    forr(i, N){
        int x = this -> fac[facility_name][facility_id].id_l[i];
        int home = this -> graph["household"][x];
        forr(j, this -> fac["household"][home].id_l.size())
            this -> ind[this -> fac["household"][home].id_l[j]].supply_level = 1;
    }      
}
    
void Env::deal_general(const string& facility_name, int facility_id){
    this -> deal_general_facility(facility_name, facility_id);
}
    
int Env::activity_level(vector<float> &action){
    return action[m_ind_action["activity"]];
}

/******* deal with simulations when a person is visiting a hospital (deprecated now, some detailed settings are not used in our paper) ********/    

void Env::deal_hospital(int facility_id){
    int N = this -> fac["hospital"][facility_id].id_l.size();
    vector<int> checklist;
    forr(i, N){
        int x = this -> fac["hospital"][facility_id].id_l[i];
        if (! this -> is_examined(x))
            checklist.push_back(x);
        if (checklist.size() >= this -> fac["hospital"][facility_id].testing_capacity) break;
    }
    forr(i, checklist.size())
        this -> ind[checklist[i]].examined = true;

    forr(i, N){
        int x = this -> fac["hospital"][facility_id].id_l[i];
        if (this -> ind[x].hospitalized)continue;
        vector<int> lst_of_receive;
        if (this -> args.hosp_policy == m_hosp_policy["receive_mild_fcfs"])
            lst_of_receive = list(m_obs["mild"], m_obs["severe"], m_obs["critical"]);
        else if (this -> args.hosp_policy == m_hosp_policy["receive_severe_fcfs"])
            lst_of_receive = list(m_obs["severe"], m_obs["critical"]);
        else if (this -> args.hosp_policy == m_hosp_policy["receive_critical_fcfs"])
            lst_of_receive = vector<int>(1, m_obs["critical"]);
        if (in(this -> ind[x].health_obs, lst_of_receive))
            if (this -> fac["hospital"][facility_id].current < this -> fac["hospital"][facility_id].hospitalized_capacity){
                this -> fac["hospital"][facility_id].current += 1;
                this -> ind[x].hospitalized = true;
                this -> ind[x].which_hospital = facility_id;
                BGL.lock();
                tot_hosp_breakdown[agenum(this->ind[x].type)]++;
	        daily_new_hosp ++;
                BGL.unlock();
	    }
    }
    this -> deal_general_facility("hospital", facility_id);
    this -> fac["hospital"][facility_id].operational_workers = 0;
    forr(i, N){
        int x = this -> fac["hospital"][facility_id].id_l[i];
        if (this -> ind[x].type == m_pt["med"] && this -> is_workable(x) && ( ! this -> ind[x].hospitalized))
            this -> fac["hospital"][facility_id].operational_workers += 1;
    }
        
}

/******* using multi-thread to accelerate initialization execute at the beginning of deal_ind_action ********/    

void mt_fac_init(Env* this_, string x, int sta, int en){
    for(int y = sta; y < en; y++)
        this_ -> fac[x][y].reset();
}
void mt_ind_init(Env* this_, int sta, int en){
    for(int i = sta;i < en; i++){
        this_ -> new_ind_hs[i] = this_ -> ind[i].health;
        this_ -> ind[i].reset();
    	this_ -> ind[i].p_infect = 0;
        if (in(this_ -> ind[i].health, list(m_hs["sym"], m_hs["asy"], m_hs["msy"], m_hs["ssy"], m_hs["csy"], m_hs["pre"]))){
            float mask_situation = (this_ -> ind[i].action[m_ind_action["protection"]] == 1) ? this_ -> args.mask_infect_rate : 1;  // it suppose "protection" = 1 means wearing mask
            float asy_situation = (this_ -> ind[i].health == m_hs["asy"]) ? this_ -> args.asy_infect_rate : 1;
            float pre_situation = (this_ -> ind[i].health == m_hs["pre"]) ? this_ -> args.pre_infect_rate : 1;
	        this_ -> ind[i].p_infect = mask_situation * asy_situation * pre_situation;
	    }
    }
}
void mt_ind_distrib(Env* this_, string i,int sta, int en, IndAction *action_ind, vector<int>* household_buyer){ // reference is forbid in multi-thread struture
        forr2(j, sta, en){
            this_ -> ind[j].has_gone[i] = false;
            if (this_ -> ind[j].health == m_hs["dea"]) continue;
	    if (this_ -> ind[j].n_qu >= 0) continue;
            bool hospitalized = this_ -> is_hospitalized(j);
            
	    if(hospitalized && i != "hospital")continue;
            //cout<<j<<endl;
	    if (i == "hospital"){
                int hospital_num = this_ -> where_hospital(j, (*action_ind)[j]);
                // print("hospital_num:", hospital_num, "tot:", this -> args.n_f[i])
                if (hospital_num != -1){
		    m[hospital_num].lock();
                    this_ -> fac[i][hospital_num].id_l.push_back(j);
		    m[hospital_num].unlock();
                    this_ -> ind[j].has_gone[i] = true;
                }
            }
            else if (i == "household"){
		int id = this_ -> graph[i][j];
		this_ -> ind[j].virtual_p_infected[0] += this_ -> fac[i][id].p;
		this_ -> ind[j].virtual_p_infected[5] += this_ -> fac[i][id].p;
		this_ -> ind[j].virtual_p_infected[6] += this_ -> fac[i][id].p2;
		m[id].lock();
                this_ -> fac[i][id].id_l.push_back(j);
		m[id].unlock();
                this_ -> ind[j].has_gone[i] = true;
            }
            else if (i == "workplace"){
   		int id = this_ -> graph[i][j];
                if (this_ -> is_working(j, (*action_ind)[j])){
		    this_ -> ind[j].virtual_p_infected[0] += this_ -> fac[i][id].p;
		    m[id].lock();
                    this_ -> fac[i][id].id_l.push_back(j);
		    m[id].unlock();
                    this_ -> ind[j].has_gone[i] = true;
                }
            }
            else if (i == "school"){
		int id = this_ -> graph[i][j];
                if (this_ -> is_schooling(j, (*action_ind)[j])){
		    this_ -> ind[j].virtual_p_infected[0] += this_ -> fac[i][id].p;
		    m[id].lock();
                    this_ -> fac[i][id].id_l.push_back(j);
		    m[id].unlock();
                    this_ -> ind[j].has_gone[i] = true;
                }
            }
            else{
                if(this_ -> graph[i][j] == -1) continue;
                if(this_ -> ind[j].type == m_pt["chd"] or this_ -> ind[j].type == m_pt["sch"]) {
			continue;
		}
		int id = this_ -> graph[i][j];
		bool is_sc = in(i, list(string("supermarket"), string("retail")));
		if (is_sc)this_ -> ind[j].virtual_p_infected[4] += this_ -> fac[i][id].p;
		else this_ -> ind[j].virtual_p_infected[m_require_activity_level[i]] += this_ -> fac[i][id].p;
                if(!(this_ -> is_outable(j) && this_ -> ind[j].action[m_ind_action["activity"]] >= m_require_activity_level[i]))continue;

                if(is_sc){
                    if(this_ -> ind[j].action[m_ind_action["market"]] == m_shop["offline"]){
			    if((*household_buyer)[this_ -> graph["household"][j]] == -1)
				    (*household_buyer)[this_ -> graph["household"][j]] = j;
		    }
                    else continue;
                }
		m[id].lock();
                this_ -> fac[i][id].id_l.push_back(j);
		m[id].unlock();
                this_ -> ind[j].has_gone[i] = true;
            }
        }
}

/******* using multi-thread to accelerate some operations on facilities ********/    

void mt_fac_deal(Env* this_, string i, int sta, int en, vector<string> l_v){
    forr2(k, sta, en){
            if (in(i,  l_v)){
                int N = this_ -> fac[i][k].adjusted_capacity;
                if (this_ -> fac[i][k].id_l.size() > N){  // kick redundant people out
                    std::random_shuffle(this_ -> fac[i][k].id_l.begin(), this_ -> fac[i][k].id_l.end());
                    BGL.lock();
                    if(i == "hospital"){
                        tot_kick += this_ -> fac[i][k].id_l.size() - N;
                    }
                    BGL.unlock();
                    forr(idx, this_ -> fac[i][k].id_l.size() - N) {
                        int x = this_->fac[i][k].id_l[idx];
                        this_->ind[x].has_gone[i] = false;  
                    }
	            }
                vector<int> new_id_l; 
                forr(l, this_ -> fac[i][k].id_l.size()){
                    int x = this_ -> fac[i][k].id_l[l];
                    if (this_ -> ind[x].has_gone[i]) new_id_l.push_back(x);
                }
                this_->fac[i][k].id_l = new_id_l;
            }

        if(i == "hospital"){
            this_ -> deal_hospital(k);
            continue;
        }
        else if(in(i, list(string("supermarket"), string("retail")))) this_ -> deal_market(i, k);
        else this_ -> deal_general(i, k);
        int N = (this_ -> fac[i][k].id_l).size();
        if(i != "workplace") continue;
        this_ -> fac[i][k].operational_workers = 0; // only applies for operational
        forr(j, N){	
            int x = this_ -> fac[i][k].id_l[j];
            if (this_ -> is_working(x, this_ -> ind[x].action))
                this_ -> fac[i][k].operational_workers += 1;
        }
    }       
}

/******* distribute each agent into facilities and calculate their prob of being infected after each individual has decided its action ********/

int Env::deal_ind_action(IndAction &action_ind){
    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
        string x = it->first;
        forr3(y, 0, this -> args.n_f[x], 1000){
            int en = min(this -> args.n_f[x], y + 1000);
            q_tr.push(thread(mt_fac_init, this, x, y, en));
        }
    }
    forr3(i, 0, this -> args.n_agent, 10000){
        int en = min(this -> args.n_agent, i + 10000);
        q_tr.push(thread(mt_ind_init, this, i, en));
    }
    mt_wait();
    this -> city.reset();
    forr(i, this -> n_agent){
        this -> ind[i].supply_level = (sqrt(this ->ind[i].supply_level) - 1.0 / 21.0) * (sqrt(this ->ind[i].supply_level) - 1.0 / 21.0);
        if (this -> ind[i].supply_level <= 0) this -> ind[i].supply_level = 0;
    }

    vector<int> household_buyer;
    household_buyer.assign(this -> args.n_f["household"], -1);
    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
        string i = it->first;
        forr3(j, 0, this -> n_agent, 10000){
            int en = min(this -> n_agent, j + 10000);
            q_tr.push(thread(mt_ind_distrib, this, i, j, en, &action_ind, &household_buyer));
        }
        mt_wait();
        vector<string> l_v = {"workplace", "hospital"};
        for(string l : l_full)l_v.push_back(l);
        forr3(k, 0, this -> args.n_f[i], 10000){
            int en = min(this -> args.n_f[i], k + 10000);
            q_tr.push(thread(mt_fac_deal, this, i, k, en, l_v));
        }       
        mt_wait();
    }
    forr(j, this -> n_agent){
        if(this -> ind[j].type != m_pt["chd"] && this -> ind[j].action[m_ind_action["market"]] == m_shop["online"] && this -> online_queue.size() < this -> args.maximum_online_request)this -> online_queue.push(j);
    } 
    forr(i, this -> args.maximum_online_request){
        if(this -> online_queue.size() == 0)break;
        int x = this -> online_queue.front();
        online_queue.pop();
        int sz = this -> fac["household"][this -> graph["household"][x]].id_l.size();
        forr(k, sz){
            int y = this -> fac["household"][this -> graph["household"][x]].id_l[k];
            this -> ind[y].supply_level = 1;
        }
    }
    for(auto it = this->graph.begin();it!=this->graph.end();it++){
        string k = it->first;
        for(auto it2 = this->fac[k].begin(); it2!=this->fac[k].end();it2++){
            Facility &f = (*it2);
            int l = f.id_l.size();
            forr(i, l){
                int x = f.id_l[i];
                if(in(this->new_ind_hs[x], l_hs_ill, sizeof(l_hs_ill)/ sizeof(l_hs_ill[0]))){
                    this -> city.n_infected[k] += 1;
                    f.n_infected += 1;
                }
                if(!this->is_examined(x)) f.unexamined += 1;
            }
        }
    }
    int newly_infected = 0;
    forr(i, this -> n_agent){
        if ((!in(this -> ind[i].health, l_hs_ill, sizeof(l_hs_ill)/ sizeof(l_hs_ill[0]))) && in(this-> new_ind_hs[i], vector<int>(l_hs_ill, l_hs_ill + sizeof(l_hs_ill) / sizeof(l_hs_ill[0]))))
            newly_infected += 1;
        if (in(this -> new_ind_hs[i], l_hs_ill, sizeof(l_hs_ill)/ sizeof(l_hs_ill[0]))){
            this -> city.n_infected_all += 1;
            this -> city.n_infected_geo[this -> graph["community"][i]] += 1;
        }
        this -> ind[i].health = this -> new_ind_hs[i];
    }

    cout<<"info of n_infected"<<endl;
    for(auto it = this->graph.begin();it!=this->graph.end();it++){
        string k = it->first;
        cout<<k<<" summed "<<this -> city.n_infected[k]<<endl;
    }
    for(auto it = this->graph.begin();it!=this->graph.end();it++){
        string k = it->first;
        cout<<k<<" new "<<this -> city.n_infected_new[k]<<endl;
    }

    return newly_infected;
}
void mt_ind_obs(Env* this_, int sta, int en, int obs_size, vector<float>* obs){
    forr2(i, sta, en){
        vector<float> v = parse_obs_ind_(this_, i);
        forr(j, obs_size)
            (*obs)[i * obs_size + j] = v[j];
    }
}

/******* government strategies including quanrantine and contact tracing in executed here ********/
/* this code also implemented partial closing (some dangerous facilities rather than all facilities of some certain types are closed), but we didn't use them */

void Env::gov_control_ind_and_fac(){
    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
	double su = 0;
        string x = it->first;
	if (x == "household")continue;
	forr(j, this -> args.n_f[x]){
	    ca_su += this -> fac[x][j].real_capacity;
	    if (D_day >= 1){
	        this -> fac[x][j].n_infected_d[D_day] = this -> fac[x][j].n_infected_d[D_day - 1];
	        this -> fac[x][j].n_infected_d2[D_day] = this -> fac[x][j].n_infected_d2[D_day - 1];
	    }
	    vector<int> new_ii;
	    for(auto y: this -> fac[x][j].infected_id){
        	if(in(this -> ind[y].health_old, list(m_hs["msy"], m_hs["ssy"], m_hs["csy"]))){
		    if (D_day >= 0)
		    this -> fac[x][j].n_infected_d2[D_day] += 1;
	        }
	        else{
		    new_ii.push_back(y);
		}
            }
	    this -> fac[x][j].infected_id = new_ii;
	    int n_ca = this -> fac[x][j].n_infected_d2[D_day] - this -> fac[x][j].n_infected_d2[max(0, D_day - 5)];
	    su += 1.0 * n_ca / this -> fac[x][j].real_capacity;
	    if (0)
	    if (n_ca > this -> fac[x][j].real_capacity * 0.01 or n_ca > 10){ // 0.1 infected -> close
		this -> fac[x][j].close = true;
		}
	    if (this -> fac[x][j].close)cl_su += this -> fac[x][j].real_capacity;
	}
	}
    forr(i, this -> n_agent){
        if (this -> ind[i].n_qu != -1 and in(this -> ind[i].health, list(m_hs["imm_a"], m_hs["imm_s"], m_hs["sus"])))this -> ind[i].n_qu += 1;
        if (this -> ind[i].n_qu >= 9) this -> ind[i].n_qu = -1;// -1 means no quarantine, n_qu means n_days after no pathogen
        if (this -> ind[i].n_qu != -1)qu_su += 1;
            if (in(this -> ind[i].health_old, list(m_hs["msy"], m_hs["ssy"], m_hs["csy"])) and this -> ind[i].n_qu == -1)
        //if (0)
        if (hpn(p_discover)){
            for(auto x:this -> ind[i].infect_id)
            if (hpn(p_contact_tracing))
                this -> ind[x].n_qu = 0;
            this -> ind[i].n_qu = 0;
            this -> ind[i].infect_id = vector<int>();
        }
    }
}

/******* the step function of a RL environment ********/
/* the pipline of the simulator is organized here */

Step_returns Env::step(AllAction actions){
    cout<<"step: "<<(this -> s_n)<<endl;
    ca_su = 0;
    cl_su = 0;
    qu_su = 0;
    if (phase == 1){
        this -> gov_control_ind_and_fac();
    }
    p_mar = 0;
    daily_new_cases = 0;
    daily_clip_ill_reward = 0;
    daily_recovery = 0;
    daily_new_hosp = 0;
    today_avg_age = 0;
    int current_severe_not_hosp = 0, current_severe = 0;
    vector<vector<int> > action_count = vector<vector<int> >{vector<int>{0, 0}, vector<int>{0, 0}, vector<int>{0, 0, 0, 0}, vector<int>{0, 0, 0}};
    forr(i, this->n_agent){
    if (phase == 0)actions.ind[i][0] = hpn(0.05);
    else {
        if(first_episode == 1){
            actions.ind[i][0] = hpn(0.7);
            if(this->s_n == stop_step - 1){first_episode = 0;}
        }
    }

    actions.ind[i][1] = !!in(this -> ind[i].health_obs, {m_obs["severe"], m_obs["critical"]});
    if (this -> ind[i].type == m_pt["sch"]){
        actions.ind[i][2] = 0;
        actions.ind[i][3] = 0;
    }
    else if (this -> ind[i].type != m_pt["chd"]){
        if(phase == 0){
            if(hpn(0.8))actions.ind[i][2] = 3;
            else if(hpn(0.33))actions.ind[i][2] = 2;
            else if(hpn(0.5))actions.ind[i][2] = 1;
            else actions.ind[i][2] = 0;
            actions.ind[i][3] = 2 * (!!hpn(1.0 / 1.0));
        }
        else if (first_episode){
            if(hpn(0.25))actions.ind[i][2] = 3;
            else if(hpn(0.3333))actions.ind[i][2] = 2;
            else if(hpn(0.5))actions.ind[i][2] = 1;
            else actions.ind[i][2] = 0;
            if(hpn(0.33))actions.ind[i][3] = 0;
            else if(hpn(0.5))actions.ind[i][3] = 1;
            else actions.ind[i][3] = 2;
        }
    }
    this -> ind[i].action = actions.ind[i];
    forr(j, 4){
        if(this->ind[i].type == m_pt["chd"] && j >= 2) continue;
        if(this->ind[i].type == m_pt["sch"] && j >= 2) continue;
        action_count[j][int(this->ind[i].action[j])]++;
    }
    }
    if (tot_new_cases >= 440)phase = 1;
    IndAction action_ind = actions.ind;
    GovAction action_gov = actions.gov;
    vector<float> t = this -> deal_gov_action(action_gov);
    int num_of_tested = t[0];
    int num_of_robots = t[1];
    float changing_penalty = t[2];
    float disinfect_cost = t[3];
    vector<int> health_old(this -> args.n_agent);
    forr(i, this -> args.n_agent)health_old[i] = this -> ind[i].health, this -> ind[i].health_old = this -> ind[i].health;
    int new_infections = this -> deal_ind_action(action_ind);
    int new_deaths = 0;
    if(D_day >= 0)D_day++;
    forr(i, this->args.n_agent){
    int old_health = this->ind[i].health;
    new_deaths += this->step_health_state(i);
    this -> ind[i].health_obs = this -> ind[i].get_obs();
    if(D_day < 0 && in(this->ind[i].health_obs, {m_obs["mild"], m_obs["severe"], m_obs["critical"], m_obs["dead"]}))
        D_day = 0;
    if(in(old_health, l_hs_ill, 8) && in(this->ind[i].health, {m_hs["sus"], m_hs["imm_a"], m_hs["imm_s"]}))
        daily_recovery++;
    }
    tot_deaths += new_deaths;
    this -> overall_observed_infected = 0, this -> overall_accurate_infected = 0;
    forr (j, this -> n_agent){
    if(in(this -> ind[j].health, l_hs_ill, sizeof(l_hs_ill)/ sizeof(l_hs_ill[0])))
        this -> overall_accurate_infected += 1;
    if(in(this -> ind[j].health_obs, l_obs_ill, sizeof(l_obs_ill) / sizeof(l_obs_ill[0])))
        this -> overall_observed_infected += 1;
    this -> city.n_p_hs[m_hs2[this -> ind[j].health]] += 1;
    }
    this -> s_n += 1;  // episode length

    // we assume all agent has the same obs_size
    vector<float> obs(obs_size * this -> n_agent);
    
    forr3(i, 0, this->n_agent, 10000){
    int en = min(this -> n_agent, i + 10000); 
    q_tr.push(thread(mt_ind_obs, this, i, en, obs_size, &obs));
    }
    mt_wait();
    Obs gov_obs = this -> parse_obs_gov();
    
    vector<float> rewards(this -> n_agent + 1, 0);
    
    forr(i, this -> n_agent){
    rewards[i] = this -> parse_reward_ind(i, this -> ind[i].action, health_old[i]);
    
    }
    t = this -> parse_reward_gov(changing_penalty, disinfect_cost, num_of_tested, num_of_robots, new_deaths, new_infections);
    
    rewards[this -> n_agent] = t[0];
    float r1 = t[1], r2 = t[2], r3 = t[3];
    forr(i, rewards.size())
    this -> sum_r[i] += rewards[i];
    bool no_infected = true;
    forr(i, this -> n_agent)
    if (in(this -> ind[i].health, l_hs_ill, sizeof(l_hs_ill) / sizeof(l_hs_ill[0]))){
        no_infected = false;
        break;
    }
    vector<int> done(this -> n_agent + 1, 0);

    vector<map<string, float> > info(this -> n_agent + 1);
    current_severe = 0;
    float avg_act_ill = 0, avg_act_noill = 0, tot_act_ill = 0, tot_act_noill = 0, avg_supply_level = 0;
    forr(i, this -> n_agent){

    if (in(this->s_n,{this -> stop_step,}))done[i] = true;
    else done[i]=false;
    if(in(this->ind[i].health, {m_hs["ssy"], m_hs["csy"]}))current_severe++;
    if(!in(this->ind[i].type, {m_pt["chd"], m_pt["sch"]})){
        float s = this->ind[i].action[m_ind_action["activity"]];
        if(in(this->ind[i].health, l_hs_ill, 8)){avg_act_ill += s;tot_act_ill++;}
        else {avg_act_noill += s;tot_act_noill++;}
    }

    if (this -> s_n >= this -> stop_step){
        info[i]["episode_r"] = this -> sum_r[i];
        this -> sum_r[i] = 0.;
    }
    avg_supply_level += this->ind[i].supply_level;
    }
    avg_supply_level /= this->n_agent;
    if(tot_act_ill == 0)tot_act_ill = 1;
    if(tot_act_noill == 0)tot_act_noill = 1;
    avg_act_ill /= tot_act_ill;
    avg_act_noill /= tot_act_noill;
    cout<<"avg act ill:"<<avg_act_ill<<endl;
    cout<<"avg act noill:"<<avg_act_noill<<endl;
    
    if (this -> s_n >= this -> stop_step) this -> s_n = 0;
    info[this -> n_agent]["revenue"] = r1 + r2 + r3; // gov info
    info[this -> n_agent]["adjusted_changing_penalty"] = this -> args.gov_reward_adjust_ratio * changing_penalty;
    info[this -> n_agent]["workplace"] = r1;
    info[this -> n_agent]["school"] = r2;
    info[this -> n_agent]["other"] = r3;
    this -> last_new_infections = new_infections;
    this -> last_new_deaths = new_deaths;
    this -> new_infection_average = 0.9 * this -> new_infection_average + 0.1 * this -> last_new_infections;
    this -> new_death_average = 0.9 * this -> new_death_average + 0.1 * this -> last_new_deaths;
    Step_returns ret;
    ret.ind_obs = obs;
    ret.gov_obs = gov_obs;
    ret.reward = rewards;
    ret.done = done;
    ret.info = info;
    ret.action_count = action_count;
    ret.graph = this -> get_graph_info();
    
    tot_new_cases += daily_new_cases;
    tot_new_hosp += daily_new_hosp;
    int avg_cnt = 0;
    double avg_r0 = 0;

    float today_avg_age_outcome = 0;
    if(daily_new_cases > 0)today_avg_age_outcome = float(today_avg_age) / (100.0 * daily_new_cases);
    if(daily_recovery > 0)avg_r0=float(daily_new_cases)/float(daily_recovery);
    tot_recovery += daily_recovery;
    fout<<D_day<<" "<<this->overall_observed_infected<<" "<<this->overall_accurate_infected<<" "<<daily_new_cases<<" "<<daily_new_hosp<<" "<<tot_new_cases<<" "<<tot_new_hosp<<" "<<current_severe<<" "<<new_deaths<<" "<<tot_deaths<<" "<<daily_recovery<<" "<<tot_recovery<<" "<<today_avg_age_outcome<<' '<<tot_chd<<' '<<tot_wk<<' '<<tot_rtr<<' '<<tot_off_hosp_death<<' '<<avg_r0<<' '<<tot_severe<<" "<<tot_kick<<' '<<tot_death_breakdown[0]<<' '<<tot_death_breakdown[1]<<' '<<tot_death_breakdown[2]<<' '<<tot_hosp_breakdown[0]<<' '<<tot_hosp_breakdown[1]<<' '<<tot_hosp_breakdown[2]<<' '<<avg_act_ill<<' '<<avg_act_noill<<' '<<daily_clip_ill_reward<<' '<<p_mar / 1180000<<' '<<avg_supply_level<<' ';
    fout<<action_count[0][0]<<' '<<action_count[0][1]<<' ';
    fout<<action_count[2][0]<<' '<<action_count[2][1]<<' '<<action_count[2][2]<<' '<<action_count[2][3]<<' ';
    fout<<action_count[3][0]<<' '<<action_count[3][1]<<' '<<action_count[3][2]<<' ';fout<<ca_su<<' '<<cl_su<<' '<<qu_su<<endl;
    /*activity and infect count*/
    int infect_age[4], activity_age[4][4], shop_age[4][3], mask_age[4]; 
    memset(infect_age, 0, sizeof(infect_age));
    memset(activity_age, 0, sizeof(activity_age));
    memset(shop_age, 0, sizeof(shop_age));
    memset(mask_age,0,sizeof(mask_age));
    int infe[this->args.n_f["community"]], act[this->args.n_f["community"]][4], act2[this->args.n_f["community"]][3], act3[this->args.n_f["community"]];
    forr(i, this->args.n_f["community"]){
        infe[i] = act[i][0] = act[i][1] = act[i][2] = act[i][3] = 0;
        act2[i][0] = act2[i][1] = act2[i][2] = act3[i] = 0;
    }
    forr(i, n_agent){
        int comm = this->graph["community"][i];
        int typ = this->ind[i].type;
        if(typ == m_pt["med"])typ = m_pt["wk"];
        if(in(this->ind[i].health, l_hs_ill, 8)){infe[comm]++;infect_age[typ]++;}
        // counting age 
        mask_age[typ] += int(this->ind[i].action[m_ind_action["protection"]]);
        if(!in(this->ind[i].type, {m_pt["chd"], m_pt["sch"]})){
            activity_age[typ][int(this->ind[i].action[m_ind_action["activity"]])]++;
            shop_age[typ][int(this->ind[i].action[m_ind_action["protection"]])]++;
        }
        // counting communities
        if(!in(this->ind[i].type, {m_pt["chd"], m_pt["sch"]})){
            act[comm][int(this->ind[i].action[m_ind_action["activity"]])]++; //   activity count
            act2[comm][int(this->ind[i].action[m_ind_action["market"]])]++;  //     market count
        }  
        act3[comm] += int(this->ind[i].action[m_ind_action["protection"]]);    // protection count
    }
    
    fout2<<D_day<<' ';
    forr(i, 4){
        fout2<<infect_age[i]<<' '<<mask_age[i]<<' ';
        forr(j, 4)fout2<<activity_age[i][j]<<' ';
        forr(j, 3)fout2<<shop_age[i][j]<<' ';
    }
    fout2<<endl;
    return ret;
}

map<string, vector<float> > Env::get_graph_info(){
    map<string, vector<float> > g_info;
    g_info["com_ca"] = vector<float>();
    g_info["com_n_infected"] = vector<float>();
    g_info["com_n_infected_new"] = vector<float>();
    g_info["com_y"] = vector<float>();
    g_info["com_x"] = vector<float>();
    forr(i, this -> args.n_f["community"]){
        g_info["com_ca"].push_back(this -> fac["community"][i].basic_capacity);
        g_info["com_n_infected"].push_back(this -> city.n_infected_geo[i]);
        g_info["com_n_infected_new"].push_back(this -> fac["community"][i].n_infected_new);
        g_info["com_y"].push_back(this -> fac["community"][i].latitude);
        g_info["com_x"].push_back(this -> fac["community"][i].longitude);
    }
    return g_info;
}

/******* the reward of each individvual is calculated here ********/

float Env::parse_reward_ind(int id, vector<float> &action, int health_old){
    
    float R = 0;
    if (this->ind[id].health == m_hs["dea"]) return R;
    
    float mask_penalty = 0.1;
    if(in(health_old, list(m_hs["sym"], m_hs["msy"], m_hs["ssy"], m_hs["csy"]))) R += mask_penalty * action[m_ind_action["protection"]]; // reward for mask
    else R -= mask_penalty * action[m_ind_action["protection"]];

    if (this -> ind[id].type != m_pt["sch"] and this -> ind[id].type != m_pt["chd"]){
    if (in(health_old, list(m_hs["sym"], m_hs["msy"], m_hs["ssy"], m_hs["csy"])))
	    R -= action[m_ind_action["activity"]]; // is the agent is infectious, he will be punished when going out.
    else
	    R += action[m_ind_action["activity"]]; // otherwise he will be rewarded. full = 3, limit = 2, local = 1, must = 0.
    }
    if ( (!in(health_old, l_hs_ill, 8)) && health_old != m_hs["imm_s"] && health_old != m_hs["imm_a"]){
        forr(j, 5){
            float mask_situation = this -> ind[id].action[m_ind_action["protection"]] ? this -> args.mask_infected_rate : 1;
            double t = (D_day - (10 + 0)) / (70.0 - 0);
                t = max(0., t);
            if(is_calibration){
                t = t * t * t * t;
                Rill = 25000 * t + 4500 * (1 - t);
            }
            else Rill = r_ill;
            if(Rill * (1 - this->ind[id].p_not_infected_fac[j]) / mask_situation >= 100){BGL.lock();daily_clip_ill_reward++;BGL.unlock();}
            R -= min(100., Rill * (1 - this -> ind[id].p_not_infected_fac[j]) / mask_situation) * mask_situation;
        }
    }
    if (this -> ind[id].type == m_pt["sch"] or this -> ind[id].type == m_pt["chd"]){
	    return R;
    }
    if(action[m_ind_action["market"]] == 2) R -= 1;
    R -= 1 / 0.58 * (1 - this -> ind[id].supply_level);
    return R;
}

/******* the reward of the government is calculated here (deprecated now) ********/

vector<float> Env::parse_reward_gov(float changing_penalty, float disinfect_cost, int num_of_tested, int num_of_robots,int new_deaths, int new_infections){
    float r = 0; // changing_penalty; efficiency. 
    r -= changing_penalty;
    float test_cost = this -> args.gov_test_cost * num_of_tested;
    float robot_cost = this -> args.gov_robot_maintainance * num_of_robots;
    float death_penalty = (2 * (this -> args.gov_death_penalty + this -> city.n_p_hs["dea"] * this -> args.gov_death_penalty_growth) - (new_deaths - 1) * this -> args.gov_death_penalty_growth) * new_deaths / 2; //growth
    float infect_penalty = (2 * (this -> args.gov_infection_penalty + overall_accurate_infected * this -> args.gov_infected_penalty_growth) - (new_infections - 1) * this -> args.gov_infected_penalty_growth) * new_infections / 2;
    r -= (test_cost + robot_cost + death_penalty + infect_penalty);
    //revenue from maintaining a normal society
    float r_workplace = 0, r_school = 0;
    forr(j, this -> args.n_f["workplace"]){
	r_workplace += this -> fac["workplace"][j].efficiency() * this -> fac["workplace"][j].revenue;
    }
    forr(j, this -> args.n_f["school"]){
        float v = float(this -> fac["school"][j].id_l.size()) / this -> fac["school"][j].basic_capacity * this -> fac["school"][j].revenue;
        if(this -> fac["school"][j].rotate) v *= 0.5;
        r_school += v;
    }
    int k = 0;
    float total_other_r = 0;
    vector<float> r_other = vector<float>(12, 0); // size of l_profitable
    for(string i: l_profitable){
        if(in(i, {"workplace", "school"})) continue;
        r_other[k] = 0;
        forr(j, this -> args.n_f[i]){
            float v = float(this -> fac[i][j].id_l.size()) / float(this -> fac[i][j].basic_capacity * this -> fac[i][j].revenue);
       	    r_other[k] += v;
        }
        total_other_r += r_other[k];
        k++;
    }
    r += r_workplace + r_school + total_other_r;
    float adjust_ratio = this -> args.gov_reward_adjust_ratio;
    return list(r * adjust_ratio, r_workplace * adjust_ratio, r_school * adjust_ratio, total_other_r * adjust_ratio);
}

/******* the observation of the government is calculated here (deprecated now) ********/

map<string, map<string, float> > Env::parse_obs_gov(){
    map<string, map<string, float> > obs;
    obs["city"] = map<string, float>();
    obs["gov_last_action"] = map<string, float>();
    obs["gov_unchanged_days"] = map<string, float>();
    obs["city"]["infected"] = this -> overall_observed_infected;
    obs["city"]["dead"] = this -> city.n_p_hs["dea"];
    obs["city"]["robots"] = 0;
    obs["city"]["current_robot"] = this -> gov_current_robot;
    obs["city"]["avg_new_infected"] = this -> new_infection_average;
    obs["city"]["avg_new_death"] = this -> new_death_average;
    
    if(this->gov_last_action.size()>0){
        int i = 0;
        for(string j: {"workplace", "school", "full", "limit", "local", "must"}){
            obs["gov_last_action"][j] = this->gov_last_action[i];
            obs["gov_unchanged_days"][j] = this->gov_unchanged_days[i];
            i++;
        }
    }
    else {
        for(string j: {"workplace", "school", "full", "limit", "local", "must"}){
            obs["gov_last_action"][j] = 0;
            obs["gov_unchanged_days"][j] = 0;
        }
    }
    for(auto it = this -> graph.begin(); it != this -> graph.end(); it ++){
        string i = it->first;
        if (i == "household")continue;
        obs[i] = map<string, float>();
        forr(j, this -> args.n_f[i]){
            obs["city"]["robots"] += this -> fac[i][j].robots;
            // This should be cafefully written! FIXME
            char buffer[10];
            sprintf(buffer, "%d\0", j);
            string str = string(buffer);
            if (i == "hospital"){
                obs[i+"_"+str]["disinfection"] = this -> is_disinfected(i, j);
                obs[i+"_"+str]["current"] = this -> fac[i][j].current;
                obs[i+"_"+str]["operational"] = this -> fac[i][j].operational_workers;
                obs[i+"_"+str]["full_workers"] = this -> fac[i][j].full_workers;
                obs[i+"_"+str]["n_infected"] = this -> fac[i][j].n_infected;
                obs[i+"_"+str]["robot"] = this -> fac[i][j].robots;
	    }
            else{
                obs[i+"_"+str]["disinfection"] = this -> is_disinfected(i, j);
                obs[i+"_"+str]["shutdown"] = this -> shutdowned(i, j);
                obs[i+"_"+str]["operational"] = this -> fac[i][j].operational_workers;
                obs[i+"_"+str]["full_workers"] = this -> fac[i][j].full_workers;
                obs[i+"_"+str]["n_infected"] = this -> fac[i][j].n_infected;
                obs[i+"_"+str]["unexamined"] = this -> fac[i][j].unexamined;
                obs[i+"_"+str]["robot"] = this -> fac[i][j].robots;
            }
        }
    }
    return obs;
}

/******* the observation of each individual is calculated here ********/

vector<float> parse_obs_ind_(Env* this_, int id){
    vector<float>obs;
    obs.reserve(obs_size);
    for(int i = 0; i < 12; i++) obs.push_back(0); // onehot(health_obs)
    obs[this_->ind[id].health_obs] = 1;//note: health obs does not have 12 dim!
    for(int i = 0; i < 2; i++) obs.push_back(0); // onehot(examined)
    obs[12 + this_->ind[id].examined] = 1;
    for(int i = 0; i < 2; i++) obs.push_back(0); // onehot(hospitalized)
    obs[14 + this_->ind[id].hospitalized] = 1;
    obs.push_back(this_ ->ind[id].supply_level); // supply_level
    obs.push_back(min(1.0, this_ -> overall_accurate_infected / 1000.0)); // city observation
    obs.push_back(this_ -> ind[id].virtual_p_infected[6]); //household
    p_mar += this_ -> ind[id].virtual_p_infected[4]; //market p
    obs.push_back(min(1.0, this_ -> overall_accurate_infected / 5000.0)); // city observation
    if (phase == 1)
	obs.push_back(0), obs.push_back(1);
    else
	obs.push_back(1), obs.push_back(0);
    double t = (D_day - 10) / 70.0;
    t = max(0., t);
    obs.push_back(t);
    obs.push_back(this_ -> ind[id].n_qu >= 0);

    forr(i, 5){
        if(information_disclosure)
            obs.push_back(min(20., this_ -> ind[id].virtual_p_infected[i] * Rill) / 20);
        else obs.push_back(0);
    }
    return obs;
}
