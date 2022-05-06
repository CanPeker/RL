import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# window size

width = 480
height = 360

# color

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# fps value

fps = 30


# main player
class Player(pygame.sprite.Sprite):

    def __init__(self):

        # init all things

        pygame.sprite.Sprite.__init__(self)

        self.boyutx = 30

        self.boyuty = 30

        self.image = pygame.Surface((30,30)) # image means player visualize

        self.image.fill(RED)

        self.rect = self.image.get_rect()


        self.rect.centerx = width/2

        self.rect.centery = height-10

        self.speedx = 0


    def update(self,action):

        # update state with actions
        print(action)
        keystate = pygame.key.get_pressed()

        if (keystate[pygame.K_LEFT] or action == 1 ):
            self.speedx = -4

        elif (keystate[pygame.K_RIGHT] or action == 0 ):
            self.speedx = 4

        else:
            self.speedx = 0


        self.rect.x += self.speedx


        if self.rect.right > width:
            self.rect.right = width
        if self.rect.left < 0:
            self.rect.left = 0



    def getCoordinates(self):

        # for being state, getting coordinates

        return (self.rect.x, self.rect.y)

    def getCoordinates_2(self):
        return self.rect.top,self.rect.right,self.rect.left

# enemy or somethings

class Enemy(pygame.sprite.Sprite):

    def __init__(self):

        pygame.sprite.Sprite.__init__(self)

        self.boyutx = 20

        self.boyuty = 20

        self.image = pygame.Surface((20,20))

        self.image.fill(BLUE)

        self.rect = self.image.get_rect()



        self.rect.x = 10

        self.rect.y = -20

        self.speedx = 0
        self.speedy = 3

    def update(self):

        self.rect.x += self.speedx
        self.rect.y += self.speedy

        if self.rect.top > height + 10:
            self.rect.x = random.randrange(0, width - self.rect.width)
            self.rect.y = -20
            self.speedy = 3

    def getCoordinates(self):
        return (self.rect.x, self.rect.y)

    def getCoordinates_2(self):
        return self.rect.bottom+3,self.rect.right,self.rect.left

# Agent


class Dql:
    def __init__(self):
        # parameter / hyperparameter
        self.state_size = 4  # distance [(playerx-m1x),(playery-m1y),(playerx-m2x),(playery-m2y)]
        self.action_size = 3  # right, left, no move

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)

        self.model = self.build_model()

    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Env

class Env(pygame.sprite.Sprite):


    def __init__(self):

        pygame.sprite.Sprite.__init__(self)
        self.players = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()

        self.player = Player()
        self.enemy = Enemy()
        self.enemy2 = Enemy()

        self.players.add(self.player)

        self.enemies.add(self.enemy)
        self.enemies.add(self.enemy2)

        self.reward = 0

        self.total_reward = 0

        self.done = False

        self.agent = Dql()

    def hits(self):

        hit = 0

        player_top, player_right, player_left = self.player.getCoordinates_2()
        enemy_bot, enemy_right, enemy_left = self.enemy.getCoordinates_2()
        enemy_bot2, enemy_right2, enemy_left2 = self.enemy2.getCoordinates_2()

        # print("player-top:",player_top,"enemy-bottom:",enemy_bot)
        # print("E-L:",enemy_left,"P-R",player_right)
        # print("P-L:",player_left,"E-R:",enemy_right)

        if (player_top + 1 == enemy_bot):

            # print("if-1")

            if (enemy_left < player_right and enemy_right > player_left):
                hit = 1

            else:
                hit = 0

        elif (player_top + 1 == enemy_bot2):
            # print("if-2")

            if (enemy_left2 < player_right and enemy_right2 > player_left):
                hit = 1
            else:
                hit = 0



        else:
            hit = 0

        return hit

    def findDistance(self, a, b):
        d = a - b
        return d

    def step(self, action):

        state_list = []

        self.players.update(action)
        self.enemies.update()

        next_state_player = self.player.getCoordinates()

        next_state_enemy = self.enemy.getCoordinates()

        next_state_enemy2 = self.enemy2.getCoordinates()

        state_list.append(self.findDistance(next_state_player[0], next_state_enemy[0]))
        state_list.append(self.findDistance(next_state_player[1], next_state_enemy[1]))

        state_list.append(self.findDistance(next_state_player[0], next_state_enemy2[0]))
        state_list.append(self.findDistance(next_state_player[1], next_state_enemy2[1]))

        return [state_list]

    def reset(self):

        self.players = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()

        self.player = Player()
        self.enemy = Enemy()
        self.enemy2 = Enemy()

        self.players.add(self.player)

        self.enemies.add(self.enemy)
        self.enemies.add(self.enemy2)

        state_list = []

        state_player = self.player.getCoordinates()

        state_enemy = self.enemy.getCoordinates()

        state_enemy2 = self.enemy2.getCoordinates()

        state_list.append(self.findDistance(state_player[0], state_enemy[0]))
        state_list.append(self.findDistance(state_player[1], state_enemy[1]))

        state_list.append(self.findDistance(state_player[0], state_enemy2[0]))
        state_list.append(self.findDistance(state_player[1], state_enemy2[1]))

        return [state_list]


    def run(self):

        state = self.reset()
        batch_size = 24
        running = True
        hit = 0



        while running:


            self.reward = 2
            clock.tick(fps)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

            hit = self.hits()


            action = self.agent.act(state)

            next_state = self.step(action)

            self.total_reward += self.reward




            if hit == 1:

                self.reward = -150
                self.total_reward += self.reward
                self.done = True
                running = False
                print("Total reward: ", self.total_reward)



            self.agent.remember(state,action,self.reward,next_state,self.done)

            self.agent.replay(batch_size)

            self.agent.adaptiveEGreedy()

            state = next_state


            screen.fill(GREEN)
            self.players.draw(screen)
            self.enemies.draw(screen)
            pygame.display.flip()

        pygame.quit()









if __name__ == "__main__":

    env = Env()
    epochs = 200

    for i in range(0,epochs):
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ACP")
        clock = pygame.time.Clock()
        env.run()













