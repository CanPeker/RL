# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:22:41 2020

@author: ACP
"""
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self, env):
        # parameter / hyperparameter
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gamma = 0.95
        self.learning_rate = 0.0001 
        
        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
        
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation = "tanh"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state, done in minibatch:
        
            if done:
                update_value = reward 
           
            else:
                next_state_value = self.model.predict(next_state)[0]
                update_value = reward + self.gamma*np.amax(next_state_value)
               
            
            # q-table = self.model.predict(state)
            # train_target(current-value) = self.model.predict(state)
            
            train_target = self.model.predict(state)
            
           
            """
            print("action:",action)
           
            print("train_target[0] = ",self.model.predict(state)[0])
           
            print("train_target[0]"+"["+str(action)+"]"+":",train_target[0][action])
            """
            
            train_target[0][action] = update_value #update value 
            
            #print("new value",train_target[0])
            
            
            self.model.fit(state,train_target, verbose = 0) #backward 
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    
    # initialize gym env and agent
    env = gym.make("CartPole-v0")
    agent = DQLAgent(env)
    
    batch_size = 16
    episodes = 100
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        
        state = np.reshape(state,[1,4])
        
        time = 0
        rewardd=0
        
        while True:
            
            # act
            action = agent.act(state) # select an action
            
            # step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state,[1,4])
            
            # remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            
            # replay
            agent.replay(batch_size)
            
            # adjust epsilon
            agent.adaptiveEGreedy()
            
             # update state
            state = next_state
            
            
            time += 1
            
            rewardd += reward
            
            env.render()
            
            if done:
                print("Episode: {}, Reward: {},time: {}".format(e,rewardd,time))
                break

import time
trained_model = agent
state = env.reset()
state = np.reshape(state, [1,4])
time_t = 0
while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1,4])
    state = next_state
    time_t += 1
    print(time_t)
    #time.sleep(0.5)
    if done:
        break
print("Done")

 

