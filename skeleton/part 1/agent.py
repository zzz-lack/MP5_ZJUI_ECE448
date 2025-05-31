import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.points = 0
        self.s = None
        self.a = 0

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = 0

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        dis_state = self.discretize_state(state)
        
        if self._train and self.s is not None:

            reward = self.reward_function(points, dead)
            # print(f"Reward: {reward}, Points: {points}, Dead: {dead}")  # Debugging

            q_value_now = self.Q[self.s][self.a]
            alpha = self.C / (self.C + self.N[self.s][self.a])

            max_future_q = max(self.Q[dis_state][i] for i in range(4))
            new_q_value =  self.gamma * max_future_q + reward
            self.Q[self.s][self.a] += alpha * (new_q_value - q_value_now)

        if dead:
            self.reset()
            return
        
        action,_= self.choose_action(dis_state)
        
        if not self._train:
            pass
        else:
            self.N[dis_state][action] += 1

        self.s = dis_state
        self.a = action
        
        self.points = points 
        # print(f"Action: {action}, State: {self.s}, Action: {self.a}")  # Debugging
        return action
    
    def discretize_state(self, state):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :return: a tuple of the discretized state
        '''
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        
        # Discretize the state based on the defined grid
        adjoining_wall_x = (snake_head_x==utils.WALL_SIZE)+2*(utils.DISPLAY_SIZE-2*utils.WALL_SIZE == snake_head_x)
        adjoining_wall_y = (snake_head_y==utils.WALL_SIZE)+2*(utils.DISPLAY_SIZE-2*utils.WALL_SIZE == snake_head_y)

        food_dir_x = (food_x < snake_head_x) + 2*(food_x > snake_head_x) 
        food_dir_y = (food_y < snake_head_y) + 2*(food_y > snake_head_y) 

        # Adjoining body segments
        adjoining_body_top = int((snake_head_x,snake_head_y -utils.GRID_SIZE) in snake_body)
        adjoining_body_bottom = int((snake_head_x,snake_head_y + utils.GRID_SIZE) in snake_body)
        adjoining_body_left = int((snake_head_x - utils.GRID_SIZE, snake_head_y) in snake_body)
        adjoining_body_right = int((snake_head_x + utils.GRID_SIZE, snake_head_y) in snake_body)

        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y,
                adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
    
    def choose_action(self, dis_state):
        '''
        :param dis_state: a tuple of the discretized state
        :return: the next action to take based on the Q-table and exploration strategy
        '''
        utilities = [-float('inf')] * 4  # Initialize utilities with negative infinity

        for action_index in range(4):
            if self.N[dis_state][action_index] >= self.Ne:
                utilities[action_index] = self.Q[dis_state][action_index]
            else:
                utilities[action_index] = 1

        max_utility = max(utilities)

        best_actions_indices = [i for i, utility in enumerate(utilities) if utility == max_utility]

        best_action = min(best_actions_indices)

        # print(f"Utilities: {utilities}, Best Action: {best_action}")  # Debugging

        return best_action, max_utility 

    def reward_function(self, points, dead):
        '''
        :param dis_state: a tuple of the discretized state
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the reward for the action taken
        '''
        reward = -0.1
        if points > self.points:
            reward = 1.0
        elif dead:
            reward = -1.0
        return reward