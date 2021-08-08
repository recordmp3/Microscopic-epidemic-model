#include"config.h"
#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdio>
#include<utility>
#define forr(i,n) for(int i = 0;i < n; i ++)
using std::cout;
using std::endl;
using std::max;
using std::ifstream;
using std::ios;
using std::pair;
using std::make_pair;
std::random_device e;
std::uniform_real_distribution<double> u(0, 1);
//useful mappings
map<int,string>m_rl = {
	{0, "health"},
	{1, "examined"},
	{2, "hospitalized"},
	{3, "type"},
        {4, "supply"}
};
map<string, int> m_hosp_policy = { //deprecated; fixed at 1 in our experiments
	{"receive_mild_fcfs", 0},
	{"receive_severe_fcfs", 1},
	{"receive_critical_fcfs", 2}
};
//health observation
map<string, int> m_obs = {
	{"healthy", 0},
	{"asymptomatic", 1},
	{"mild", 2},
	{"severe", 3},
	{"critical", 4},
	{"immune", 5},
	{"dead", 6}
};
map<int, string> m_obs2 = {
	{0, "healthy"},
	{1, "asymptomatic"},
	{2, "mild"},
	{3, "severe"},
	{4, "critical"},
	{5, "immune"}, 
	{6, "dead"}
};
//health state
map<string, int> m_hs = {
	{"sus", 0},
	{"inc", 1},
	{"ina", 2},
	{"sym", 3},
	{"asy", 4},
	{"msy", 5},
	{"ssy", 6},
	{"csy", 7},
	{"imm_a", 8},
	{"imm_s", 9},
	{"dea", 10},
	{"pre", 11}
};
map<int, string> m_hs2 = {
    {0, "sus"},
    {1, "inc"}, 
    {2, "ina"},
    {3, "sym"},
    {4, "asy"},
    {5, "msy"},
    {6, "ssy"},
    {7, "csy"},
    {8, "imm_a"},
    {9, "imm_s"},
    {10, "dea"},
    {11, "pre"}
};
map<string, int> m_al = {
    {"must", 0},
    {"local", 1},
    {"lim", 2},
    {"full", 3}
};
//age group
map<string, int> m_pt = {
    {"chd", 0},
    {"sch", 1},
    {"wk", 2},
    {"rtr", 3},
    {"med", 4}
};
map<int, string> m_apt = {
    {0, "chd"},
    {1, "sch"},
    {2, "wk"},
    {3, "rtr"},
    {4, "med"}
};
//action index
map<string, int> m_ind_action = {
    {"protection", 0},
    {"hospital", 1},
    {"activity", 2},
    {"market", 3}
};

map<string, int>m_shop = {
    {"no", 0},
    {"online",1},
    {"offline", 2}
};
//facility index
map<float, string>m_fac = {
    {0, "workplace"},
    {1, "school"},
    {2, "hospital"},
    {3, "household"},
    {4, "stadium"},
    {5, "cinema"},
    {6, "theatre"},
    {7, "library"},
    {8, "museum"},
    {9, "gym"},
    {10, "retail"},
    {11, "community"},
    {12, "restaurant"},
    {13, "supermarket"}
};

map<string, float>m_fac2 = {
    {"workplace", 0},
    {"school", 1},
    {"hospital", 2},
    {"household", 3},
    {"stadium", 4},
    {"cinema", 5},
    {"theatre", 6},
    {"library", 7},
    {"museum", 8},
    {"gym", 9},
    {"retail", 10},
    {"community", 11},
    {"restaurant", 12},
    {"supermarket", 13}
};
//MinAct
map<string, int>m_require_activity_level ={
    {"supermarket", 0}, // also including household, workplace, school
    {"retail", 0},
    {"community", 1},
    {"library", 2},
    {"museum", 2},
    {"gym", 2},
    {"stadium", 3},
    {"cinema", 3},
    {"restaurant", 3},
    {"theatre", 3},
};

int l_hs_ill[8] = {m_hs["inc"], m_hs["pre"], m_hs["sym"], m_hs["msy"], m_hs["ssy"], m_hs["csy"], m_hs["asy"], m_hs["ina"]}; 
int l_obs_ill[4] = {m_obs["asymptomatic"], m_obs["mild"], m_obs["severe"], m_obs["critical"]};

string l_must_only[2] = {"supermarket", "retail"};
string l_local_only[1] = {"community"};
string l_limit_only[3] = {"library", "museum", "gym"};
string l_full_only[4] = {"stadium", "cinema", "restaurant", "theatre"};

string l_must[2] = {"supermarket", "retail"};
string l_local[3] = {"retail", "community", "supermarket"};
string l_limit[6] = {"library", "museum", "gym", "retail", "community", "supermarket"};
string l_full[10] = {"stadium", "cinema", "theatre", "library", "museum", "gym", "retail", "community", "restaurant", "supermarket"};
string l_facility[14] = {"workplace", "school", "hospital", "household", "stadium", "cinema", "theatre", "library", "museum", "gym", "retail", "community", "restaurant", "supermarket"};
string l_profitable[11] = {"workplace", "school", "stadium", "cinema", "theatre", "library", "museum", "gym", "retail", "restaurant", "supermarket"};
string l_robotize[3] = {"hospital", "workplace", "retail"};
string gov_fac_lst[13] = {"workplace", "school", "household", "stadium", "cinema", "theatre", "library", "museum", "gym", "retail", "community", "restaurant", "supermarket"};
vector<string> linear_l{"household", "school", "hospital", "workplace"};
bool hpn(double p){ //true with probability of p
    return u(e) < p;
}

// calculate distance based on longtitude & latitude
float calc_dist_earth(float j1, float w1, float j2, float w2){
    float R = 6371;
    float phi1 = w1 * M_PI / 180.0;
    float phi2 = w2 * M_PI / 180.0;
    float delta_phi = (phi2 - phi1) * M_PI / 180.0;
    float delta_lambda = (j2 - j1) / 180.0;
    float a = sin(delta_phi / 2) * sin(delta_phi / 2) + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) * sin(delta_lambda / 2);
    float c = 2 * atan2(sqrt(a), sqrt(1 - a));
    float d = R * c;
    return d;
}

Args::Args(){ 
    this->action_size = {
        {"chd", 2},
        {"sch", 4},
        {"wk", 4},
        {"rtr", 4},
        {"med", 4},
    };
    this->n_agent = 0;
    this->action_repeat = 1;
    this->stop_step = 80;
    //this->hospitalized_capacity = 8;
    //this->hospital_testing_capacity = 200;
    this->hosp_policy = m_hosp_policy["receive_severe_fcfs"]; // fixed
    this->recreational_reward = 0.2; //deprecated
    this->community_reward = 0.04;   //deprecated
    memset(this->agent_count, 0, sizeof(this->agent_count));
    this->maximum_online_request = 17000; // roughly 1/70 population
    this->gov_disinfect_cost = 1; // this->gov_* and this->meta_* are all deprecated
    this->gov_test_cost = 0.005;  
    this->gov_robot_maintainance = 0.01;
    this->gov_death_penalty = 10;
    this->gov_infection_penalty = 1;
    this->gov_reward_adjust_ratio = 0.1;
    this->gov_action_size = 14;
    this->gov_maximum_robot = 30;
    this->gov_infected_penalty_growth = 0.02;
    this->gov_death_penalty_growth = 2.333;   
    //infectivity-related parameters
    this->meta_death_penalty = 1;//deprecated 
    this->meta_infection_penalty = 0.01;//deprecated
    this->mask_infect_rate = 0.4;
    this->mask_infected_rate = 0.4;
    this->n_disinfected_last = 1;//deprecated 
    this->disinfected_rate = 0.5;//deprecated
    this->p_inc2pre = 1.0/3;
    this->p_ina2asy = 1.0/2;
    this->p_pre2sym = 1.0/2;
    this->p_hos = 1.0/1.2;
    this->p_rec_sym = 1.0/(10-1.2);
    this->p_sym2sev[0] = 0.021;
    this->p_sym2sev[1] = 0.085;
    this->p_sym2sev[2] = 0.333;
    this->p_sev2cri[0] = 0.014 * 1.25;
    this->p_sev2cri[1] = 0.125 * 1.25;
    this->p_sev2cri[2] = 0.618 * 1.25;
    this->p_cri2dea[0] = 1;
    this->p_cri2dea[1] = 1;
    this->p_cri2dea[2] = 1;
    this->p_sev2rec = 1.0/10;
    this->p_cri2rec = 1.0/13;
    this->p_sev2cri_nhos[0] = 0.6;
    this->p_sev2cri_nhos[1] = 0.8;
    this->p_sev2cri_nhos[2] = 1;
    this->p_rec_asy = 1.0/4.4;

    this->p_deimm_asy = 0.009;
    this->p_deimm_sym = 0.0025;
    this->asy_pop_rate = 0.25;
    this->asy_infect_rate = 0.31;
    this->pre_infect_rate = 0.12;
    this->work_reward = 0.2;//deprecated
    this->hpn = 0; // possiblity of going into asymptomatic
    for(string i:l_facility){this->graph[i] = vector<int>();this->n_f[i]=0;}
}
void Args::init(){
    //read graph
    this -> alpha = 0.75; // 1.0 when no inflation of infection rate in large places
    this -> beta = 500 * 0.178 * 0.178;
    this -> n_f["community"] = 0;
    string l_input[13] = {"household", "workplace", "school", "retail", "gym", "library", "cinema", "museum", "restaurant", "supermarket", "stadium", "theatre", "community"};
    int tot_hosp_med[105];
    memset(tot_hosp_med, 0, sizeof(tot_hosp_med));
    int n_agent;
    ifstream fin("graph_sorted.txt",ios::in), fin2("param_list.txt",ios::in);
    //input the agents. Each agent is described by a 15-dimensional vector, wher the first dimension is age and the rest 14 are its affiliations to the facilities -1 represents for no affiliation.
    fin>>this -> n_agent;
    n_agent = this -> n_agent;
    forr(i, n_agent){
        int age, typ, tmp;
        this -> graph["hospital"].push_back(-1);
        fin>>age;
        this->age.push_back(age);
        if(age < 10)typ = m_pt["chd"];
        else if(age < 18)typ = m_pt["sch"];
        else if(age < 65)typ = m_pt["wk"];
        else typ = m_pt["rtr"];
        this -> type.push_back(typ);
        this -> agent_count[typ]++; // agent_count["wk"] = actual wk + med.
        
        for(string j: l_input){
            fin>>tmp;
            this->graph[j].push_back(tmp);
        }     
        fin>>tmp;
        if(tmp != -1) {this -> type[i] = m_pt["med"], this -> graph["hospital"][i] = tmp; tot_hosp_med[tmp]++;}
    }
    map<string, float> m_infect_rate = {{"household", 0.23}, {"school", 0.14}, {"workplace", 0.21}, {"hospital", 0}, {"community", 0.0075}, {"supermarket", 0.09}, {"retail", 0.09}, {"library", 0.12}, {"museum", 0.12}, {"gym", 0.15}, {"restaurant", 0.21}, {"cinema", 0.42}, {"theatre",0.42}, {"stadium", 0.42}};
    cout<<"input stage 1 complete!"<<endl;
    //input each type of facility.
    for(string j: l_input){
        fin>>this->n_f[j];
        string name = j, typ = j;
        int capacity;
        float infect_rate = m_infect_rate[typ];
        float longtitude = 0, latitude = 0, revenue, ind_reward, freq;
	//calculation + correction of the visiting frequency 
        if(typ == "library") freq = 10.5 / 365.0;
        else if(typ == "cinema") freq = 5.3 / 365.0 / 0.59;
        else if(typ == "stadium") freq = 4.7 / 365.0 / 0.17;
        else if(typ == "theatre") freq = 3.8 / 365.0 / 0.35;
        else if(typ == "museum") freq = 2.5 / 365.0 / 0.24;
        else if(typ == "restaurant") freq = 4.2 / 7.0;
        else if(typ == "gym") freq = 0.47;
        else if(typ == "supermarket") freq = 1.0 / 5.554;
        else if(typ == "retail") freq = 1.0 / 6.0;
        else if(typ == "workplace" || typ == "school") freq = 5.0 / 7.0; // workplace, school
        else freq = 1; // household, hospital, community 
	int ca_sum = 0;
        forr(i, this->n_f[j]){
            float a, b, c, d;
            fin >> a >> b >> c >> d;
            longtitude = c, latitude = d;
            capacity = floor(a);
            revenue = b, name = "name";
            this->name_f[j].push_back(name);
            this->info_f[j].push_back(map<string, float>());
            this->info_f[j][i]["type"] = m_fac2[j];
            this->info_f[j][i]["infect_rate"] = infect_rate;
	    if (typ == "workplace" or typ == "school" or typ == "household" or typ == "supermarket" or typ == "retail") 
            	this->info_f[j][i]["capacity"] = capacity;
	    else this->info_f[j][i]["capacity"] = capacity / freq;
	    this -> info_f[j][i]["real_capacity"] = capacity;
            ca_sum += this -> info_f[j][i]["capacity"];
	    this->info_f[j][i]["avg_freq"] = freq;
            this->info_f[j][i]["revenue"] = revenue;
            this->info_f[j][i]["longtitude"] = longtitude;
            this->info_f[j][i]["latitude"] = latitude;
        }
	if(!in(typ, linear_l)){

		float ca_sum2 = 0, ca_sum3 = 0;
		forr(i, this->n_f[j]){
			if (this -> info_f[j][i]["real_capacity"] < 0)
				this -> info_f[j][i]["real_capacity"] = 0;
			ca_sum2 += this -> info_f[j][i]["real_capacity"];
		}
		float average_capacity = ca_sum2 / this -> n_f[j];
		forr(i, this->n_f[j]){
			this -> info_f[j][i]["infect_rate"] *= pow(this -> info_f[j][i]["real_capacity"] / average_capacity, 0);//, 1 - this -> alpha);
			ca_sum3 += this -> info_f[j][i]["real_capacity"]* this -> info_f[j][i]["infect_rate"];
		}
	}
    }
    cout<<"input stage 2 complete!"<<endl;
    fin>>this->n_f["hospital"];
    forr(i, this->n_f["hospital"]){
        float a, b, c, d, e, f;
        fin>> a >> b >> c >> d >> e >> f;// doctors, capacity(beds), <deprecated>, <deprecated>, longtitude and latitude.
        float longtitude = e, latitude = f;
        string name = "name";
        float capacity = b, infect_rate = 0.5;
        this -> name_f["hospital"].push_back(name);
        this -> info_f["hospital"].push_back(map<string, float>());
        this -> info_f["hospital"][i]["type"] = m_fac2["hospital"];
        this -> info_f["hospital"][i]["infect_rate"] = m_infect_rate["hospital"];
        this -> info_f["hospital"][i]["capacity"] = b + tot_hosp_med[i];
        this -> info_f["hospital"][i]["doctors"] = a;
        this -> info_f["hospital"][i]["longtitude"] = longtitude;
        this -> info_f["hospital"][i]["latitude"] = latitude;
        this -> info_f["hospital"][i]["avg_freq"] = 1;
    }
    cout<<"input stage 3 complete!"<<endl;
    this -> hospital_list = vector<vector<int> >();
    forr(i, n_agent){ //build list of hospital.
        if(i%100000==0)cout<<"stage 4 progress:"<<i<<' '<<n_agent<<endl;
        this -> hospital_list.push_back(vector<int>());
        vector<pair<float, float> > lst = vector<pair<float, float> >();
        forr(j, this -> n_f["hospital"]){
            if(this -> graph["household"][i] == -1){cout<<"Error household!"<<endl;exit(0);}
            float j1 = this -> info_f["household"][this -> graph["household"][i]]["longtitude"];
            float w1 = this -> info_f["household"][this -> graph["household"][i]]["latitude"];
            float j2 = this -> info_f["hospital"][j]["longtitude"];
            float w2 = this -> info_f["hospital"][j]["latitude"];
            float dist = calc_dist_earth(j1, w1, j2, w2);
            lst.push_back(make_pair(dist, j));
        }
        sort(lst.begin(), lst.end());
        int sz = lst.size();
        forr(j, sz)
            this -> hospital_list[i].push_back(lst[j].second);
    } 
    cout<<this->agent_count[m_pt["chd"]]<<' '<<this->n_agent - this->agent_count[m_pt["chd"]]<<endl;
    cout<<"Input ends!"<<endl;
    fin.close();
    fin2.close();
}
