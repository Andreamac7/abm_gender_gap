# Group members:
# Antonella Buccione - 3015999
# Jacopo Bugini - 3027525
# Andrea Maccarrone - 3013402
# Sebastiano Moro - 3017824

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np
from mesa.datacollection import DataCollector

class Worker_training(Agent):
    def __init__(self, unique_id, model, level=0, age=23, education=max(np.random.randn(1)[0]+2.5,0),total_tenure=0):
        super().__init__(unique_id, model)
        self.gender = np.random.choice(['M','F'], p=([1/2]*2))
        self.level = level
        self.education = education
        self.age = age
        self.skills = np.random.randn(1)[0]*1.5
        self.level_tenure = 0
        self.total_tenure = total_tenure
        self.trials = 0
        self.aspiration = 0
        self.aspiration_boost = 0
        self.value = 0
    
    def update(self):
        # age
        self.age+=1/12
        
        # trials
        if self.level==0 and self.trials<4:
            self.trials+=1/12
        if self.trials>=4:
            return
        
        # tenure update
        if self.level>0:
            self.total_tenure+=1/12
            self.level_tenure+=1/12 
        
        # aspiration & value
        if self.gender=='F':
            self.aspiration = 3*(self.level+1)*(self.education) + np.log(self.level_tenure+1)/2 - self.trials - 3*self.model.gap_perceived/(self.level+1)
            self.value = 3*(self.level+1)*(self.education + self.skills) + np.log(self.level_tenure+1)/2 - max(0, (self.age-40))
        else:
            self.aspiration = 3*(self.level+1)*(self.education) + np.log(self.level_tenure+1)/2 - self.trials 
            self.value = 3*(self.level+1)*(self.education + self.skills) + np.log(self.level_tenure+1)/2 - max(0, (self.age-40))
   
    def change_aspiration(self):
        # Neighbors
        if np.random.rand()<1/12:
            neighbors = [neighbor for neighbor in self.model.grid.neighbor_iter(self.pos) if neighbor.aspiration >= self.aspiration]
            self.tot_neig = len(list(neighbors))
            if self.tot_neig > 0:
                female_neig=len([neighbor for neighbor in self.model.grid.neighbor_iter(self.pos) if neighbor.gender=='F' and neighbor.aspiration >= self.aspiration])
                if self.gender=='M':
                    average_aspiration=np.mean([neighbor.aspiration for neighbor in self.model.grid.neighbor_iter(self.pos) if neighbor.aspiration >= self.aspiration])
                    self.aspiration= self.aspiration*self.model.alpha + average_aspiration*(1-self.model.alpha)
                elif self.gender=='F'and female_neig>0:
                    average_aspiration=np.mean([neighbor.aspiration for neighbor in self.model.grid.neighbor_iter(self.pos) if neighbor.gender=='F' and neighbor.aspiration >= self.aspiration])
                    self.aspiration= self.aspiration*self.model.alpha + average_aspiration*(1-self.model.alpha)
    
    def training(self):
        if np.random.rand()<(1/24) and self.age<35 and self.gender=='M':
            self.education*=max(1.01,np.random.randn(1)[0]*0.2+1)
        elif np.random.rand()<(1/36) and self.age<35 and self.gender=='F':
            self.education*=max(1.01,np.random.randn(1)[0]*0.2+1)   

    def motivational_event(self):
        if np.random.rand()<(1/6) and self.age <= 35 and self.gender=='F':
            aspirations=sorted([agent.aspiration for agent in self.model.schedule.agents if agent.gender == 'F'],reverse=True)[:3]
            if len(aspirations) > 0: 
                if np.mean(aspirations) > self.aspiration:
                    best_aspiration=np.mean(aspirations)
                    self.aspiration = self.aspiration*self.model.alpha + best_aspiration*(1-self.model.alpha)
                   
    def maternity(self):
        if np.random.rand()<(3/1000) and self.gender=='F'and self.age<35 and self.level<2:
            self.trials=4
            
        
    def seek_job(self):
        if self.trials>=4:
            return
        possible_jobs=[j for j in self.model.jobschedule.agents]
        desired_jobs=[job for job in possible_jobs if job.level==self.level+1 \
                      and self.level_tenure >= job.tenure_required \
                      and self.education >= job.education_required \
                      and self.aspiration*0.75<=job.wage \
                      and self.aspiration*1.25>=job.wage]
        if len(desired_jobs)>0:
            other_agent = self.random.choice(desired_jobs)
            other_agent.candidates.append(self)
        
    def step(self):
        self.update()
        self.maternity()
        self.change_aspiration()
        self.training()
        self.seek_job()

class JobOffer_training(Agent):
    def __init__(self, unique_id, model,p=[0.73/0.9,0.12/0.9,0.05/0.9]):
        super().__init__(unique_id, model)
        self.level=np.random.choice([i+1 for i in range(3)], p=p) 
        self.candidates=[]
        
        # education
        if self.level==1:
            self.education_required=np.random.choice([2,3], p=[0.2,0.8])
        elif self.level == 2:
            self.education_required=np.random.choice([3,4], p=[0.3,0.7])
        elif self.level == 3:
            self.education_required=np.random.choice([4,5], p=[0.5,0.5])
            
        self.ranking = np.random.randn(1)[0]*1.5
        
        # tenure
        self.tenure_required=0
        if self.level == 2:
            self.tenure_required=12
        elif self.level == 3:
            self.tenure_required=10
        
        self.wage = 3*self.level * (self.education_required + self.ranking) +  np.log(self.tenure_required+1)/2
            
    def choose_candidate(self):
        
        if len(self.candidates)>0:
            best_candidate=self.candidates[0]
            for cand in self.candidates:
                if cand.value>best_candidate.value:
                    best_candidate=cand
            best_candidate.level+=1
            best_candidate.level_tenure=0
    
    def step(self):
        self.choose_candidate()
        
def average_aspiration_M(model):
    return np.mean([agent.aspiration for agent in model.schedule.agents if agent.gender=='M'])

def average_aspiration_F(model):
    return np.mean([agent.aspiration for agent in model.schedule.agents if agent.gender=='F'])

def average_level_M(model):
    return np.mean([agent.level for agent in model.schedule.agents if agent.gender=='M'])

def average_level_F(model):
    return np.mean([agent.level for agent in model.schedule.agents if agent.gender=='F'])

def average_agents_M(model):
    return sum([1/model.num_agents for agent in model.schedule.agents if agent.gender=='M'])

def average_agents_F(model):
    return sum([1/model.num_agents for agent in model.schedule.agents if agent.gender=='F'])

def average_tenure_M(model):
    return np.mean([agent.level_tenure for agent in model.schedule.agents if agent.gender=='M'])

def average_tenure_F(model):
    return np.mean([agent.level_tenure for agent in model.schedule.agents if agent.gender=='F'])

def average_age_F(model):
    return np.mean([agent.age for agent in model.schedule.agents if agent.gender=='F'])

def average_age_M(model):
    return np.mean([agent.age for agent in model.schedule.agents if agent.gender=='M'])

def average_skills_F(model):
    return np.mean([agent.skills for agent in model.schedule.agents if agent.gender=='F'])

def average_skills_M(model):
    return np.mean([agent.skills for agent in model.schedule.agents if agent.gender=='M'])

def average_value_F(model):
    return np.mean([agent.value for agent in model.schedule.agents if agent.gender == 'F'])

def average_value_M(model):
    return np.mean([agent.value for agent in model.schedule.agents if agent.gender == 'M'])

def average_value(model):
    return np.mean([agent.value for agent in model.schedule.agents])

def male_level_distribution(model):
    tot=sum([1 for agent in model.schedule.agents if agent.gender=='M'])
    return {i:sum([1/tot for agent in model.schedule.agents if agent.gender=='M' and agent.level==i]) for i in range(4)}

def female_level_distribution(model):
    tot=sum([1 for agent in model.schedule.agents if agent.gender=='F'])
    return {i:sum([1/tot for agent in model.schedule.agents if agent.gender=='F' and agent.level==i]) for i in range(4)}

def male_skill_distribution(model):
    return {i:np.mean([agent.skills for agent in model.schedule.agents if agent.gender=='M' and agent.level==i]) for i in range(4)}

def female_skill_distribution(model):
    return {i:np.mean([agent.skills for agent in model.schedule.agents if agent.gender=='F' and agent.level==i]) for i in range(4)}

def male_value_distribution(model):
    return {i:np.mean([agent.value for agent in model.schedule.agents if agent.gender=='M' and agent.level==i]) for i in range(4)}

def female_value_distribution(model):
    return {i:np.mean([agent.value for agent in model.schedule.agents if agent.gender=='F' and agent.level==i]) for i in range(4)}
        
class LabourMarket_training(Model):
    def __init__(self, N, width, height, M=0.4):
        self.num_agents = N
        self.total_agents_count = N
        self.gap_perceived=0
        self.alpha=0.6
        self.p_initial=[0.1,0.73,0.12,0.05]
        self.num_jobs=int(N*M)
        self.grid = MultiGrid(width, height, True)
        self.schedule=RandomActivation(self)

        
        # Create agents
        for i in range(self.num_agents):
            level = np.random.choice([0, 1, 2, 3], p=self.p_initial)
            education = max(0,np.random.randn(1)[0]+level+1)
            age = max(23,np.random.randn(1)[0]*12+44)
            total_tenure=max(0,np.random.randn(1)[0]*10+25)

            a = Worker_training(i, self, level, age, education,total_tenure)
            self.schedule.add(a)
            
            if a.gender == 'M':
                a.level = np.random.choice([0, 1, 2, 3], p=[0.09,0.71,0.14,0.06])
            elif a.gender == 'F':
                a.level = np.random.choice([0, 1, 2, 3], p=[0.12,0.76,0.10,0.02])

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        
        self.datacollector = DataCollector(
        model_reporters={'aspiration_M': average_aspiration_M,'aspiration_F': average_aspiration_F,
                         'level_M': average_level_M,'level_F': average_level_F,
                         'agents_M': average_agents_M,'agents_F': average_agents_F,
                         'tenure_M': average_tenure_M,'tenure_F': average_tenure_F,
                         'age_M': average_age_M,'age_F': average_age_F,
                         'skills_M': average_skills_M,'skills_F': average_skills_F,
                         'average_value':average_value,
                         'average_value_M':average_value_M,'average_value_F':average_value_F,
                         'male_level_distribution': male_level_distribution,
                         'female_level_distribution': female_level_distribution,
                         'male_skill_distribution': male_skill_distribution,
                         'female_skill_distribution': female_skill_distribution,
                         'male_value_distribution': male_value_distribution,
                         'female_value_distribution': female_value_distribution}) 
        
    def update_gap_perceived(self):
        self.gap_perceived = average_level_M(self)-average_level_F(self)
    
    def add_agents(self):
        new_entry = len([agent for agent in self.schedule.agents if agent.trials >= 4 \
                    or (agent.total_tenure + agent.age) >= 100 \
                    or agent.age > 60 \
                    or agent.total_tenure > 40])
        
        if new_entry > 0:        
            for i in range(new_entry):
                self.total_agents_count+=1
                a = Worker_training(self.total_agents_count, self)
                self.schedule.add(a)

                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                self.grid.place_agent(a, (x, y))
                self.num_agents+=1

    def remove_agents(self):            
        old_agents=[agent for agent in self.schedule.agents if agent.trials >= 4 \
                    or (agent.total_tenure + agent.age) >= 100 \
                    or agent.age > 60 \
                    or agent.total_tenure > 40] 
        
        if len(old_agents) > 0:
            for i in old_agents:
                self.schedule.remove(i)
                self.num_agents-=1         
            
            
    def step(self):
        self.add_agents()
        self.remove_agents()
        self.jobschedule = RandomActivation(self)
        for i in range(self.num_jobs):
            j = JobOffer_training(i, self)
            self.jobschedule.add(j)
        self.update_gap_perceived()
        self.schedule.step()
        self.jobschedule.step()
        self.datacollector.collect(self)
        