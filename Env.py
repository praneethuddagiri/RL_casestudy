# Import routines

import numpy as np
import math
import random
from itertools import permutations


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + list(permutations([i for i in range(m)], 2))
        self.action_space = [list(i) for i in self.action_space]
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    # Use this function if you are using architecture-1 
    def state_encod_arch1(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0 for _ in range(m+t+d)]
        #location
        state_encod[state[0]] = 1
        #time
        state_encod[m+state[1]] = 1
        #day
        state_encod[m+t+state[2]] = 1
        
        #set action vector if pickup loc is not 0
        if(action[0] != 0):
            state_encod[m+t+d+action[0]] = 1
        if(action[1] != 0):
            state_encod[m+t+d+m+action[1]] = 1
        
        return state_encod
    
    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for _ in range(m+t+d)]
        #location
        state_encod[state[0]] = 1
        #time
        state_encod[m+state[1]] = 1
        #day
        state_encod[m+t+state[2]] = 1

        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # [0,0] is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        return possible_actions_index,actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        if(action == [0,0]): #no booking accepted
            reward = -C
            #print("No ride")
        else:
            if curr_loc == pickup_loc: #pickup request is from present driver's location
                ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
                reward = (R-C)*ride_time
                #print("same loc ride")
            else: #current and pickup locs are different
                pickup_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
                
                new_time,new_day = self.get_updt_time_day(curr_time, curr_day, pickup_time)
                
                ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
                
                reward = (R-C)*ride_time - C*pickup_time
                #print("diff loc ride")
                
        #print("from env.py reward is: ",reward)
        return int(reward)
    
    
    def get_updt_time_day(self, time, day, ride_duration):
        """Takes present time, present day and ride duration to give end time and end day"""
        
        #ride duration is float
        ride_duration = int(ride_duration)
        #check if day overflow happens
        if time + ride_duration < 24:
            time  = time + ride_duration
        else: #overflow
            num_days = (time + ride_duration) // 24
            time = (time + ride_duration) % 24
            
            #handle wraparound of day
            day = (day + num_days) % 7
            
        return time,day

    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        #required to decide episode end
        total_time = 0
        
        #list copy
        next_state = [i for i in state]
        if action != [0,0]:
            next_state[0] = drop_loc
            
            if  curr_loc == pickup_loc: #pickup request is from present driver's location
                ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
                new_time,new_day = self.get_updt_time_day(curr_time, curr_day, ride_time)
                
                total_time = ride_time
            else: #current and pickup locs are different
                pickup_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
                new_time,new_day = self.get_updt_time_day(curr_time, curr_day, pickup_time)
                
                ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
                new_time,new_day = self.get_updt_time_day(new_time, new_day, ride_time)
                
                total_time = ride_time + pickup_time
        else: #no ride accepted - increment by one time unit
            total_time = 1
            new_time,new_day = self.get_updt_time_day(curr_time, curr_day, 1)
        
        next_state[1] = new_time
        next_state[2] = new_day
        return total_time, next_state
    

    def step(self, state, action, Time_matrix):
        """Environment step - returns next_state, reward and time taken for completion of action"""
        time_taken, next_state = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(state, action, Time_matrix)
        
        return next_state, reward, time_taken


    def reset(self):
        return self.action_space, self.state_space, self.state_init
