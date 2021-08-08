#include"config.h"
#ifndef FACILITY_H
#define FACILITY_H
#include<utility>
using std::pair;
class Facility{
    public:
    string type, name;
    int n_infected, n_infected_new;
    int basic_capacity, adjusted_capacity, real_capacity, relative_capacity, n_d, current;
    bool disinfected, shut, rotate;
    float longitude, latitude;
    float hospitalized_capacity, testing_capacity;
    vector<int> id_l;
    int robots, unexamined;
    Facility();
    Facility(map<string, float> info, string name);
    void reset();
    float efficiency();
    int operational_workers, full_workers;
    float revenue, ind_reward, avg_frequency, infect_rate;
    float p;
    float p2;
    vector<int> n_infected_d, n_infected_d2;
    vector<int> infected_id;
    bool close;
};

#endif
