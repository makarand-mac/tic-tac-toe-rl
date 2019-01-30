'''
   Copyright 2017 Neil Slater

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numpy as np
import csv
import random
from itertools import groupby
import pickle

class TicTacToeGame():
    def __init__(self):
        self.state = '         '
        self.player = 'X'
        self.winner = None

    def allowed_moves(self):
        states = []
        for i in range(len(self.state)):
            if self.state[i] == ' ':
                states.append(self.state[:i] + self.player + self.state[i+1:])
        return states

    def make_move(self, next_state):
        if self.winner:
            raise(Exception("Game already completed, cannot make another move!"))
        if not self.__valid_move(next_state):
            raise(Exception("Cannot make move {} to {} for player {}".format(
                    self.state, next_state, self.player)))

        self.state = next_state
        self.winner = self.predict_winner(self.state)
        if self.winner:
            self.player = None
        elif self.player == 'X':
            self.player = 'O'
        else:
            self.player = 'X'

    def playable(self):
        return ( (not self.winner) and any(self.allowed_moves()) )

    def predict_winner(self, state):
        lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        winner = None
        for line in lines:
            line_state = state[line[0]] + state[line[1]] + state[line[2]]
            if line_state == 'XXX':
                winner = 'X'
            elif line_state == 'OOO':
                winner = 'O'
        return winner

    def __valid_move(self, next_state):
        allowed_moves = self.allowed_moves()
        if any(state == next_state for state in allowed_moves):
            return True
        return False

    def print_board(self):
        s = self.state
        print('     {} | {} | {}'.format(s[0],s[1],s[2]))
        print('    -----------')
        print('     {} | {} | {}'.format(s[3],s[4],s[5]))
        print('    -----------')
        print('     {} | {} | {}'.format(s[6],s[7],s[8]))

class Agent():
    def __init__(self, game_class, epsilon=0.1, alpha=0.5, value_player='X'):
        self.V = dict()
        self.NewGame = game_class
        self.epsilon = epsilon
        self.alpha = alpha
        self.value_player = value_player

    def state_value(self, game_state):
        return self.V.get(game_state, 0.0)

    def learn_game(self, num_episodes=1000):
        for episode in range(num_episodes):
            self.learn_from_episode()

    def learn_from_episode(self):
        game = self.NewGame()
        _, move = self.learn_select_move(game)
        while move:
            move = self.learn_from_move(game, move)

    def learn_from_move(self, game, move):
        game.make_move(move)
        r = self.__reward(game)
        td_target = r
        next_state_value = 0.0
        selected_next_move = None
        if game.playable():
            best_next_move, selected_next_move = self.learn_select_move(game)
            next_state_value = self.state_value(best_next_move)
        current_state_value = self.state_value(move)
        td_target = r + next_state_value
        self.V[move] = current_state_value + self.alpha * (td_target - current_state_value)
        return selected_next_move

    def learn_select_move(self, game):
        allowed_state_values = self.__state_values( game.allowed_moves() )
        if game.player == self.value_player:
            best_move = self.__argmax_V(allowed_state_values)
        else:
            best_move = self.__argmin_V(allowed_state_values)

        selected_move = best_move
        if random.random() < self.epsilon:
            selected_move = self.__random_V(allowed_state_values)

        return (best_move, selected_move)

    def play_select_move(self, game):
        allowed_state_values = self.__state_values( game.allowed_moves() )
        if game.player == self.value_player:
            return self.__argmax_V(allowed_state_values)
        else:
            return self.__argmin_V(allowed_state_values)

    def demo_game(self, verbose=False):
        game = self.NewGame()
        t = 0
        while game.playable():
            if verbose:
                print(" \nTurn {}\n".format(t))
                game.print_board()
            move = self.play_select_move(game)
            game.make_move(move)
            t += 1
        if verbose:
            print(" \nTurn {}\n".format(t))
            game.print_board()
        if game.winner:
            if verbose:
                print("\n{} is the winner!".format(game.winner))
            return game.winner
        else:
            if verbose:
                print("\nIt's a draw!")
            return '-'

    def interactive_game(self, agent_player='X'):
        game = self.NewGame()
        t = 0
        while game.playable():
            print(" \nTurn {}\n".format(t))
            game.print_board()
            if game.player == agent_player:
                move = self.play_select_move(game)
                game.make_move(move)
            else:
                move = self.__request_human_move(game)
                game.make_move(move)
            t += 1

        print(" \nTurn {}\n".format(t))
        game.print_board()

        if game.winner:
            print("\n{} is the winner!".format(game.winner))
            return game.winner
        print("\nIt's a draw!")
        return '-'

    def round_V(self):
        # After training, this makes action selection random from equally-good choices
        for k in self.V.keys():
            self.V[k] = round(self.V[k],1)

    def save_v_table(self):
        with open('state_values.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State', 'Value'])
            all_states = list(self.V.keys())
            all_states.sort()
            for state in all_states:
                writer.writerow([state, self.V[state]])

    def save_model(self, name='new'):
        with open('obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(self.V, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name='new'):
        with open('obj/' + name + '.pkl', 'rb') as f:
            self.V = pickle.load(f)

    def __state_values(self, game_states):
        return dict((state, self.state_value(state)) for state in game_states)

    def __argmax_V(self, state_values):
        max_V = max(state_values.values())
        chosen_state = random.choice([state for state, v in state_values.items() if v == max_V])
        return chosen_state

    def __argmin_V(self, state_values):
        min_V = min(state_values.values())
        chosen_state = random.choice([state for state, v in state_values.items() if v == min_V])
        return chosen_state

    def __random_V(self, state_values):
        return random.choice(list(state_values.keys()))

    def __reward(self, game):
        if game.winner == self.value_player:
            return 1.0
        elif game.winner:
            return -1.0
        else:
            return 0.0

    def __request_human_move(self, game):
        allowed_moves = [i+1 for i in range(9) if game.state[i] == ' ']
        human_move = None
        while not human_move:
            idx = int(input('Choose move for {}, from {} : '.format(game.player, allowed_moves)))
            if any([i==idx for i in allowed_moves]):
                human_move = game.state[:idx-1] + game.player + game.state[idx:]
        return human_move

def demo_game_stats(agent):
    results = [agent.demo_game() for i in range(10000)]
    game_stats = {k: results.count(k)/100 for k in ['X', 'O', '-']}
    print("    percentage results: {}".format(game_stats))

if __name__ == '__main__':
    agent = Agent(TicTacToeGame, epsilon = 0.2, alpha = 1.0)
    agent.load_model('3x315lmodel0.2eps')
    # print("Before learning:")

    # agent.learn_game(1000)
    # print("1000 learning games:")

    # agent.learn_game(4000)
    # print("5000 learning games:")

    # agent.learn_game(5000)
    # print("10000 learning games:")

    # agent.learn_game(10000)
    # print("20000 learning games:")

    # agent.learn_game(10000)
    # print("30000 learning games:")

    # agent.learn_game(20000)
    # print("50000 learning games:")

    # agent.learn_game(50000)
    # print("100000 learning games:")

    # agent.learn_game(50000)
    # print("150000 learning games:")

    # agent.learn_game(50000)
    # print("200000 learning games:")

    # agent.learn_game(50000)
    # print("250000 learning games:")

    # agent.learn_game(50000)
    # print("300000 learning games:")

    # agent.learn_game(50000)
    # print("350000 learning games:")

    # agent.learn_game(50000)
    # print("400000 learning games:")

    # agent.learn_game(50000)
    # print("450000 learning games:")

    # agent.learn_game(50000)
    # print("500000 learning games:")
    demo_game_stats(agent)

    # agent.learn_game(100000)
    # print("600000 learning games:")
    # demo_game_stats(agent)

    # agent.learn_game(200000)
    # print("800000 learning games:")
    # demo_game_stats(agent)

    # agent.learn_game(200000)
    # print("1000000 learning games:")
    # demo_game_stats(agent)

    # agent.learn_game(500000)
    # print("1500000 learning games:")
    # demo_game_stats(agent)

    agent.save_model(name='3x315lmodel0.2eps')

    agent.interactive_game(agent_player='O')

    # agent.round_V()
    # agent.save_v_table()


