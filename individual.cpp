  #include"individual.h"
  #include<iostream>
  using std::cout;
  using std::endl;
  void Individual::init(int type, int age){ 
      this->has_ill=false; // has the individual fallen ill before (not equivalent to immune since immunity will fade.)
      this->type = type;
      this->health = 0;//0=suspectible
      this->examined = this->hospitalized = false;
      this->which_hospital = -1; //None in Python
      this->health_obs = 0;
      this->health_old = 0;
      this->n_qu = -1;
      this->has_gone.clear();
      this->p_not_infected = 1;
      this->get_infected_now = false;
      this->supply_level = 1;
      this->infect_id = vector<int>();//for quarantine; who is infected by this agent?
      this->age=age;
      for(int i = 0;i<12;i++){
      	  this->virtual_p_infected[i] = 0; 
	  this->p_not_infected_fac[i] = 1;
      }
      this->has_examined_ill = false; // has the individual been examined ill?
      if (type == m_pt["chd"])this -> infected_rate = 0.4;
      else if (type == m_pt["sch"])this -> infected_rate = 0.38;
      else if (type == m_pt["wk"])this -> infected_rate = 0.8175;
      else if (type == m_pt["rtr"])this -> infected_rate = 0.81;
      else if (type == m_pt["med"])this -> infected_rate = 0.8175; //! now I assume med is the same as wk 
      else{
	      cout<<"bug in individual.cpp"<<endl;
	      cout<<type<<endl;
	      exit(0);
      }
  }  
  void Individual::reset(){
      this->p_not_infected = 1;
      this->get_infected_now = false;
      for(int i = 0;i<12;i++){
	  this->p_not_infected_fac[i] = 1;
      	  this->virtual_p_infected[i] = 0; //for risk calculation
      }
  }
  int Individual::get_obs(){
      if(this->health == m_hs["inc"] || this->health == m_hs["pre"] || this->health == m_hs["asy"] || this->health == m_hs["sus"] || \
      this->health == m_hs["imm_a"] || this->health == m_hs["imm_s"] || this->health == m_hs["ina"])
        return m_obs["healthy"];
      else if(this->health == m_hs["sym"] || this->health == m_hs["msy"])
        return m_obs["mild"];
      else if(this->health == m_hs["ssy"]) return m_obs["severe"];
      else if(this->health == m_hs["csy"]) return m_obs["critical"];
      else return m_obs["dead"];
  }
