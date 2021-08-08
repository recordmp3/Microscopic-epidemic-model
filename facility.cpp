#include"facility.h"
#include<iostream>
#include<utility>
using std::max;
using std::min;
using std::cout;
using std::endl;
Facility::Facility(map<string, float> info, string name){
    this->type = m_fac[info["type"]];
    this->name = name;
    this->longitude = info["longtitude"];
    this->latitude = info["latitude"];
    this->disinfected = false;//deprecated.
    this->n_infected = 0;
    this->avg_frequency = info["avg_freq"];
    this->infect_rate = info["infect_rate"];
    this->n_d = -1, this->current = 0;
    this->shut = false;//true when capacity = 0.
    this->id_l.clear();
    this->robots = this->unexamined = 0;//deprecated.
    this->rotate = false;//for schools.
    this->operational_workers = this->full_workers = 0;
    this->revenue = 0;//deprecated.
    this->p = 0;
    this->p2 = 0;
    this->infected_id = vector<int>();
    this->real_capacity = info["real_capacity"];
    if (this -> type != "household")
    {
    	this->n_infected_d = vector<int>(200); // n_infecte for each day
    	this->n_infected_d2 = vector<int>(200); // n_infected known by gov for each day
    }
    this -> close = false;
    if(this -> type != "hospital"){
        this->basic_capacity = info["capacity"];
        this->adjusted_capacity = this->basic_capacity;
    }
    else{
        this->testing_capacity = 5 * info["doctors"];
        this->hospitalized_capacity = info["capacity"];
	this->basic_capacity = this->adjusted_capacity = this->hospitalized_capacity;
    }
}
Facility::Facility(){}
void Facility::reset(){
    this->disinfected = false;//deprecated.
    this->n_infected = this->n_infected_new = 0; // no need for resetting the adjusted capacity.
    this->id_l.clear();
}
float Facility::efficiency(){ // used for calculating government utilities; deprecated.
    if(this->full_workers == 0) return 0;
    return min((float)1, max(float(this->operational_workers + min(this->robots, this -> operational_workers/3)) / this->full_workers, (float)0.05));
}
