import time
import setup_path
import airsim
import sys
import pprint
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn import preprocessing
from scipy.interpolate import interp1d
import random
import keras
from keras.models import load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D


class Dql:

    def __init__(self):

        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.975
        #self.model = self.build_model()
        self.model = self.load_model("D:/Pycharm/RL/model-i9.h5")
        self.memory = deque(maxlen=5000)

    def remember(self,state,action,reward,next_state,done,action_number):
        self.memory.append((state,action,reward,next_state,done,action_number))


    def inter(self, list, maks, minn):



        maxx = max(list)
        minnn = min(list)
        list2 = []



        for i in range(0, len(list)):
            m = interp1d([maxx, minnn], [maks, minn])
            x = list[i]
            y = float(m(x))
            y = round(y, 4)
            # print(round(y, 4))
            list2.append(y)

        return list2

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=2, activation="tanh"))
        model.add(Dense(32, input_dim=32, activation="relu"))
        model.add(Dense(4, activation="softmax"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, model_name):
        model = self.build_model()
        model.load_weights(model_name)
        return model


    def action(self,state):

        state = np.array(state)
        action_1 = []



        if random.uniform(0,1) <= self.epsilon:

            #print("random")
            for i in range(0,4):
                action = random.randrange(-5,5)
                action_1.append(action)

            action_numberr = np.argmax(np.array(action_1))

            #print("random")
            return action_1,action_numberr

        else:
        

            #print("ysa")

            action = self.model.predict(state)

            action = self.inter(action[0],5,-5)

            action_number = np.argmax(action[0])


            return action,action_number

    def train(self,batch_size):

        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory,batch_size)

        #print(minibatch)

        for state,action,reward,next_state,done,action_number in minibatch:

            state = np.array(state)
            next_state = np.array(next_state)


            #print("current_state_value", current_state_value)

            if done:
                update_value = reward
            else:
                next_state_value = self.model.predict(next_state)[0]
                #print("next_state_values:", next_state_value)

                update_value = reward + (self.gamma * (np.amax(next_state_value)))
            #print("update_value:", update_value)

            current_state_value = self.model.predict(state)
            current_state_value[0][action_number] = update_value
            self.model.fit(state, current_state_value, verbose=0)

            #print("current_state:", current_state_value[0])

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




class Drone:

    def __init__(self,vehicle_name):

        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.state = self.client.getMultirotorState()
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        #print(self.state)

    def ddtodms(self, dd):
        mnt, sec = divmod(dd * 3600, 60)
        deg, mnt = divmod(mnt, 60)

        return deg, mnt, round(sec, 1)

    def gps_data(self):
        self.state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        return self.state.gps_location.latitude,self.state.gps_location.longitude,self.state.gps_location.altitude

    def move(self,x,y,z,yaw,t):
        self.client.moveByVelocityZAsync(x, y, z, t, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                         airsim.YawMode(False, yaw),vehicle_name=self.vehicle_name).join()




    def poss(self):
        #print("rondom poss...")
        x = random.randrange(-5,5)
        y = random.randrange(-5,5)
        z = random.randrange(-3,3)

        self.client.moveToPositionAsync(x, y, z, 5,vehicle_name=self.vehicle_name).join()
        #print("rondom poss")

    def layer(self):

        state = self.client.getMultirotorState(vehicle_name="Drone1")
        vehicle_y, vehicle_x = state.gps_location.latitude, state.gps_location.longitude

        state2 = self.client.getMultirotorState(vehicle_name="Drone2")
        vehicle2_y, vehicle2_x = state2.gps_location.latitude, state2.gps_location.longitude

        _, _, y1s = self.ddtodms(vehicle_y)
        _, _, y2s = self.ddtodms(vehicle2_y)

        _, _, x1s = self.ddtodms(vehicle_x)
        _, _, x2s = self.ddtodms(vehicle2_x)

        #print(x1s,x2s,y1s,y2s)

        xx=0
        yy=0
        z=0

        if(y1s>31 or y2s>31):
            yy = 1
            z=1
            print("Layer!")

        if (y1s < 27 or y2s < 27):
            yy = 2
            z = 1
            print("Layer!")


        if (x1s > 37 or x2s > 37):
            xx = 1
            z = 1
            print("Layer!")

        if (x1s < 33 or x2s < 33):
            xx = 2
            z = 1
            print("Layer!")

        return xx,yy,z



    def update(self,action):

        #print("update:",action)
        #xx,yy = self.layer()


        """
        if (xx == 1):
            self.move(0,-5,0,0,2)
            print("x right layer")
        if (xx == 2):
            self.move(0, 5, 0, 0, 2)
            print("x left layer")

        if (yy == 1):
            self.move(-5,0,0,0,2)
            print("y up layer")
        if (yy == 2):
            self.move(5, 0, 0, 0, 2)
            print("y down layer")

        """



        x = action[0]
        y = action[1]
        yaw = action[2]
        t = abs(action[3])


        self.move(x,y,1,yaw,t)

    def getState(self):
        lat,long,_ = self.gps_data()
        return lat,long




class Env:

    def __init__(self):
        self.vehicle = Drone("Drone1")
        self.vehicle2 = Drone("Drone2")
        self.agent = Dql()
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.episode = 0
        self.x = 0
        self.i = 0
    def calculate_state(self,a,b):
        return a-b



    def step(self,action):

        self.vehicle.update(action)

        action_2 = []

        for i in range(0, 4):
            actionn = random.randint(-5, 5)
            action_2.append(actionn)


        self.vehicle2.update(action_2)


        y1,x1 = self.vehicle.getState()
        y2, x2 = self.vehicle2.getState()

        y = self.calculate_state(y2,y1)
        x = self.calculate_state(x2,x1)

        next_state = []

        next_state.append(y)
        next_state.append(x)


        return [next_state]

    def reset(self):
        #self.vehicle = Drone()

        self.vehicle.poss()
        self.vehicle2.poss()
        self.done = False
        self.reward = 0
        self.total_reward = 0
        y1, x1 = self.vehicle.getState()
        y2, x2 = self.vehicle2.getState()

        y = self.calculate_state(y2, y1)
        x = self.calculate_state(x2, x1)

        state = []

        state.append(y)
        state.append(x)

        return [state]

    def ddtodms(self,dd):
        mnt, sec = divmod(dd * 3600, 60)
        deg, mnt = divmod(mnt, 60)

        return deg, mnt, round(sec, 1)


    def hit(self):

        hits = 0

        vehicle = self.vehicle.getState()
        vehicle2 = self.vehicle2.getState()

        vehicle_x,vehicle_y = vehicle[1],vehicle[0]
        vehicle2_x, vehicle2_y = vehicle2[1], vehicle2[0]

        y1d,y1m,y1s = self.ddtodms(vehicle_y)
        y2d, y2m, y2s = self.ddtodms(vehicle2_y)

        x1d, x1m, x1s = self.ddtodms(vehicle_x)
        x2d, x2m, x2s = self.ddtodms(vehicle2_x)

        #print("vehicle1:",y1d,y1m,y1s,x1d,x1m,x1s,"vehicle2:", y2d, y2m, y2s, x2d, x2m, x2s)

        #print("state1:",x2s-0.1,x1s,x2s+0.1,"state2:",y2s-0.3,y1s,y1s-0.3)

        if(x2s-0.2 < x1s < x2s+0.2 and y2s+0.2 > y1s > y1s-0.2):
            hits = 1

        #if(vehicle2_y-0.0000010<vehicle_y<vehicle2_y+0.0000010):
            #hits = 1

        else:
            hits = 0

        return hits


    def run(self):

        print("Starting Point Of "+str(self.episode))

        state = self.reset()

        running = True

        batch_size = 8

        self.episode += 1


        if(self.episode%50 == 0):
            self.i +=1
            self.agent.save_model("D:\Pycharm\RL\model-i-"+str(self.i)+".h5")




        time1 = time.time()

        while running:

            self.reward = -100

            action,action_number = self.agent.action(state)


            next_state = self.step(action)

            _, _, z = self.vehicle.layer()

            hits = self.hit()

            timey = time.time()

            if(timey-time1>300):
                self.reward = -1000
                self.total_reward += self.reward
                running = False
                done = True


            if(hits==1 or z==1):

                if(z==1):
                    self.reward = -1000
                    self.total_reward += self.reward
                    running = False
                    done = True
                else:

                    timey = time.time()
                    if(timey-time1<50):
                        self.reward = 3000
                        self.total_reward += self.reward
                        running = False
                        done = True

                    self.reward = 1000
                    self.total_reward += self.reward
                    running = False
                    done = True



            timex=time.time()

            if(timex-time1>200):
                self.reward = -500
                self.total_reward += self.reward



            self.total_reward += self.reward

            self.agent.remember(state,action,self.reward,next_state,self.done,action_number)

            self.agent.train(batch_size)

            self.agent.adaptiveEGreedy()

            state = next_state


        time2 = time.time()
        #print("state:",state)
        #print("action:",action)
        #print("next_state:",next_state)

        print("Episode:",self.episode,"total_reward:",self.total_reward,"Time:",time2-time1)





env = Env()

epochs = 1000



for i in range(0,epochs):

    env.run()

#env.hit()


