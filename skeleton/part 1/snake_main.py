import pygame
from pygame.locals import *
import argparse

from agent import Agent
from snake import SnakeEnv
import utils
import time

class Application:
    def __init__(self, args):
        self.args = args
        self.env = SnakeEnv(args.snake_head_x, args.snake_head_y, args.food_x, args.food_y)
        self.agent = Agent(self.env.get_actions(), args.Ne, args.C, args.gamma)
        
    def execute(self):
        if not self.args.human:
            if self.args.train_eps != 0:
                self.train()
            self.test()
        self.show_games()

    def train(self):
        print("Train Phase:")
        self.agent.train()
        window = self.args.window
        self.points_results = []
        first_eat = True
        start = time.time()

        for game in range(1, self.args.train_eps + 1):
            print(f"Training Episode: {game}/{self.args.train_eps}")
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            while not dead:
                state, points, dead = self.env.step(action)

                # For debug convenience, you can check if your Q-table mathches ours for given setting of parameters
                # (see Debug Convenience part on homework 4 web page)
                if first_eat and points == 1:
                    self.agent.save_model(utils.CHECKPOINT)
                    first_eat = False

                action = self.agent.act(state, points, dead)

    
            points = self.env.get_points()
            self.points_results.append(points)
            if game % self.args.window == 0:
                print(
                    "Games:", len(self.points_results) - window, "-", len(self.points_results), 
                    "Points (Average:", sum(self.points_results[-window:])/window,
                    "Max:", max(self.points_results[-window:]),
                    "Min:", min(self.points_results[-window:]),")",
                )
            self.env.reset()
        print("Training takes", time.time() - start, "seconds")
        self.agent.save_model(self.args.model_name)

    def test(self):
        print("Test Phase:")
        self.agent.eval()
        self.agent.load_model(self.args.model_name)
        points_results = []
        start = time.time()

        for game in range(1, self.args.test_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            while not dead:
                state, points, dead = self.env.step(action)
                action = self.agent.act(state, points, dead)
            points = self.env.get_points()
            points_results.append(points)
            self.env.reset()

        print("Testing takes", time.time() - start, "seconds")
        print("Number of Games:", len(points_results))
        print("Average Points:", sum(points_results)/len(points_results))
        print("Max Points:", max(points_results))
        print("Min Points:", min(points_results))

    def show_games(self):
        print("Display Games")
        self.env.display()
        pygame.event.pump()
        self.agent.eval()
        points_results = []
        end = False
        for game in range(1, self.args.show_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            count = 0
            while not dead:
                count +=1
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_ESCAPE] or self.check_quit():
                    end = True
                    break
                state, points, dead = self.env.step(action)
                # Qlearning agent
                if not self.args.human:
                    action = self.agent.act(state, points, dead)
                # for human player
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                action = 0
                            elif event.key == pygame.K_DOWN:
                                action = 1
                            elif event.key == pygame.K_LEFT:
                                action = 2
                            elif event.key == pygame.K_RIGHT:
                                action = 3
            if end:
                break
            self.env.reset()
            points_results.append(points)
            print("Game:", str(game)+"/"+str(self.args.show_eps), "Points:", points)
        if len(points_results) == 0:
            return
        print("Average Points:", sum(points_results)/len(points_results))

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
                
            
def main():
    parser = argparse.ArgumentParser(description='CS440 MP4 Snake')

    parser.add_argument('--human', default = False, action="store_true",
                        help='making the game human playable - default False')

    parser.add_argument('--model_name', dest="model_name", type=str, default="q_agent.npy",
                        help='name of model to save if training or to load if evaluating - default q_agent')

    parser.add_argument('--train_episodes', dest="train_eps", type=int, default=10000,
                        help='number of training episodes - default 10000')

    parser.add_argument('--test_episodes', dest="test_eps", type=int, default=1000,
                        help='number of testing episodes - default 1000')

    parser.add_argument('--show_episodes', dest="show_eps", type=int, default=10,
                        help='number of displayed episodes - default 10')

    parser.add_argument('--window', dest="window", type=int, default=100,
                        help='number of episodes to keep running stats for during training - default 100')

    parser.add_argument('--Ne', dest="Ne", type=int, default=40,
                        help='the Ne parameter used in exploration function - default 40')

    parser.add_argument('--C', dest="C", type=int, default=40,
                        help='the C parameter used in learning rate - default 40')

    parser.add_argument('--gamma', dest="gamma", type=float, default=0.7,
                        help='the gamma paramter used in learning rate - default 0.7')

    parser.add_argument('--snake_head_x', dest="snake_head_x", type=int, default=200,
                        help='initialized x position of snake head  - default 200')

    parser.add_argument('--snake_head_y', dest="snake_head_y", type=int, default=200,
                        help='initialized y position of snake head  - default 200')

    parser.add_argument('--food_x', dest="food_x", type=int, default=80,
                        help='initialized x position of food  - default 80')

    parser.add_argument('--food_y', dest="food_y", type=int, default=80,
                        help='initialized y position of food  - default 80')


    args = parser.parse_args()
    app = Application(args)
    app.execute()

if __name__ == "__main__":
    main()
