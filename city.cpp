#include"city.h"
//City object initialization and reset function.
City::City(){
    this->n_p_hs.clear();
    this->n_infected.clear();
    this->n_infected_new.clear();
    this->n_infected_all = 0;
    for(auto it = m_hs.begin();it!=m_hs.end();it++){
        this -> n_p_hs[it->first] = 0;
    }
}
//called every day(step) to renew information.
void City::reset(){
    this->n_infected_all = 0;
    for(auto it = m_hs.begin();it!=m_hs.end();it++)
        this->n_p_hs[it->first] = 0;
    for(string x: l_facility)
    {
        this->n_infected_new[x] = this -> n_infected[x] = 0;
    }
    this -> n_infected_geo = this -> n_infected_new_geo = vector<int>(400, 0);
} 
