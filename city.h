#include"config.h"
#include<map>
#include<vector>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#ifndef CITY_H
#define CITY_H
class City{
    public:
    Args args;
    map<string, int> n_p_hs;//counter: number of people in each health state
    map<string, int> n_infected_new, n_infected;
    int n_infected_all;
    vector<int> n_infected_geo; //deprecated; used for transmitting cases in each community from C++ to python.
    vector<int> n_infected_new_geo; // deprecated
    void reset();
    City(); 
};
#endif 
