# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:04:14 2023

@author: Dell
"""
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction , Point 
from model import Linear_QNet , QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # to control the randomeness of the next action
        self.gamma = 0.9  # discount rate 
        self.memory = deque(maxlen = MAX_MEMORY) #when the length of the deque starts increasing more than the MAXMEMORY then the elements starts poping from the left
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model , lr =LR , gamma = self.gamma)
        
        #model and trainer 
    
    
    #From the below function we get state of the game from the function
    def get_state(self, game):
        head = game.snake[0]
        
        #The below are the points around the head
        
        point_l = Point(head.x-20 , head.y)
        point_r = Point(head.x+20 , head.y)
        point_u = Point(head.x , head.y-20)
        point_d = Point(head.x , head.y+20)
        
        #The below are the boolean variables which denotes the direction the snake is moving and only one of them is 1 and rest are 0
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
    
        state = [
            #Danger Straight 
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            #Danger right 
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            #Danger left
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #Food location 
            game.food.x <  game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y ]
        
        return np.array(state, dtype = int)
    
    #The below function is used to store the state , action taken from that state , reward it got and calculate the next state and whether game over or not
    def remember(self, state , action , reward , next_state , done):
        self.memory.append((state , action ,reward , next_state , done)) #pop left if the maximum memory is reached
        
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory , BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory
        
        #Mini samples consists of tupules which consists of the states , actions , rewards , next states and dones , the below line seperates all the states , actions , rewards , nextsteps , dones in individual lists
        states , actions , rewards , next_states , dones =  zip(*mini_sample)
        self.trainer.train_step(states , actions , rewards , next_states , dones)
            
    
    def train_short_memory(self , state , action , reward , next_state , done):
        self.trainer.train_step(state , action , reward , next_state , done)
    
    def get_action(self, state):
        #randome moves : tradeoff exploration and exploitation
        self.epsilon = 80 - self.n_games 
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get the current state of the game
        state_old = agent.get_state(game)
        
        #we will get action from get action state 
        final_move = agent.get_action(state_old)
        
        #perform the move and get the new state 
        reward, done , score = game.play_step(final_move)
        
        #get the newstate after performing the action
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old , final_move, reward , state_new, done)
        
        #remember 
        agent.remember(state_old, final_move , reward , state_new, done)
        
        if done:
            #train the long memory and plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score 
                agent.model.save()
            print('Game' , agent.n_games, 'Score' , score , 'Record:' , record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/ agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores , plot_mean_scores)
            
if __name__ == '__main__':
    train()