#include"config.h"
#ifndef IND_H
#define IND_H
class Individual{
    public:
        void init(int type, int age);
	void reset();
        int type, health, which_hospital, health_obs, age;
	double p_not_infected, p_infect, infected_rate;
        float supply_level;
	bool get_infected_now, has_examined_ill, has_ill;
        bool examined, hospitalized;
	double virtual_p_infected[12], p_not_infected_fac[12];
        vector<float> action;
	vector<int> infect_id;
        map<string, int> has_gone;
        int get_obs();
	int health_old;
	int n_qu;//remaining days of quarantine. -1 = not under quarantine; >0 is the days it has been quarantined.
};
#endif  
