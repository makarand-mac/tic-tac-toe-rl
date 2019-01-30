"""
   Copyright 2019 Makarand Deshpande

   https://opensource.org/licenses/MIT
"""

import random
import pickle


class TTTGame:
    def __init__(self):
        self.state = "         "
        self.player = "X"
        self.winner = None

    def allowed_moves(self):
        states = []
        for i in range(len(self.state)):
            if self.state[i] == " ":
                states.append(self.state[:i] + self.player + self.state[i + 1 :])
        return states

    def make_move(self, next_state):
        if self.winner:
            raise (Exception("Game already completed, cannot make another move!"))
        if not self.__valid_move(next_state):
            raise (
                Exception(
                    "Cannot make move {} to {} for player {}".format(
                        self.state, next_state, self.player
                    )
                )
            )

        self.state = next_state
        self.winner = self.predict_winner(self.state)
        if self.winner:
            self.player = None
        elif self.player == "X":
            self.player = "O"
        else:
            self.player = "X"

    def playable(self):
        return (not self.winner) and any(self.allowed_moves())

    def predict_winner(self, state):
        lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        winner = None
        for line in lines:
            line_state = state[line[0]] + state[line[1]] + state[line[2]]
            if line_state == "XXX":
                winner = "X"
            elif line_state == "OOO":
                winner = "O"
        return winner

    def __valid_move(self, next_state):
        allowed_moves = self.allowed_moves()
        if any(state == next_state for state in allowed_moves):
            return True
        return False

    def print_board(self):
        s = self.state
        print("     {} | {} | {}".format(s[0], s[1], s[2]))
        print("    -----------")
        print("     {} | {} | {}".format(s[3], s[4], s[5]))
        print("    -----------")
        print("     {} | {} | {}".format(s[6], s[7], s[8]))
        print("    -------------\n\n")


class RLTicTacToeAgent:
    def __init__(self, game_class, epsilon=0.1, alpha=0.5, value_player="X"):
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
        r = self.get_reward(game)
        td_target = r
        next_state_value = 0.0
        selected_next_move = None
        if game.playable():
            best_next_move, selected_next_move = self.learn_select_move(game)
            next_state_value = self.state_value(best_next_move)
        current_state_value = self.state_value(move)
        td_target = r + next_state_value
        self.V[move] = current_state_value + self.alpha * (
            td_target - current_state_value
        )
        return selected_next_move

    def learn_select_move(self, game):
        allowed_state_values = self.__state_values(game.allowed_moves())
        if game.player == self.value_player:
            best_move = self.__argmax_V(allowed_state_values)
        else:
            best_move = self.__argmin_V(allowed_state_values)

        selected_move = best_move
        if random.random() < self.epsilon:
            selected_move = self.__random_V(allowed_state_values)

        return (best_move, selected_move)

    def play_select_move(self, game):
        allowed_state_values = self.__state_values(game.allowed_moves())
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
            return "-"

    def interactive_game(self, agent_player="X"):
        game = self.NewGame()
        t = 0
        while game.playable():
            print(" \nTurn {}\n".format(t))
            game.print_board()
            if game.player == agent_player:
                move = self.play_select_move(game)
                game.make_move(move)
            else:
                move = self.get_human_move(game)
                game.make_move(move)
            t += 1

        print(" \nTurn {}\n".format(t))
        game.print_board()

        if game.winner:
            print("\n{} is the winner!".format(game.winner))
            return game.winner
        print("\nIt's a draw!")
        return "-"

    def round_V(self):
        # After training, this makes action selection random from equally-good choices
        for k in self.V.keys():
            self.V[k] = round(self.V[k], 1)

    def __state_values(self, game_states):
        return dict((state, self.state_value(state)) for state in game_states)

    def __argmax_V(self, state_values):
        max_V = max(state_values.values())
        chosen_state = random.choice(
            [state for state, v in state_values.items() if v == max_V]
        )
        return chosen_state

    def __argmin_V(self, state_values):
        min_V = min(state_values.values())
        chosen_state = random.choice(
            [state for state, v in state_values.items() if v == min_V]
        )
        return chosen_state

    def __random_V(self, state_values):
        return random.choice(list(state_values.keys()))

    def get_reward(self, game: TTTGame):
        if game.winner == self.value_player:
            return 1.0
        elif game.winner:
            return -1.0
        else:
            return 0.0

    def play_move(self, game: TTTGame):
        legal_states = self.__state_values(game.allowed_moves())
        if game.player == self.value_player:
            return self.__argmax_V(legal_states)
        else:
            return self.__argmin_V(legal_states)

    def demo(self):
        game: TTTGame = self.NewGame()
        t = 0

        while game.game_in_progress():
            move = self.play_move(game)
            game.move(move)

            t += 1

        if game.winner:
            return game.winner
        else:
            return "-"

    def human_game(self, agent="X"):

        game: TTTGame = self.NewGame()

        t = 0

        while game.playable():
            print('Turn {}'.format(t))
            game.print_board()
            if game.player == agent:
                move = self.play_move(game)
                game.make_move(move)
            else:
                move = self.get_human_move(game)
                game.make_move(move)
            t += 1

        game.print_board()
        if game.winner:
            print("\n{} is the winner!".format(game.winner))
            return game.winner
        print("\nIt's a draw!")
        return "-"

    def save_model(self, name="primary_model"):
        file_name = "models/" + name + ".pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.V, f, pickle.HIGHEST_PROTOCOL)
        return file_name

    def load_model(self, name="primary_model"):
        with open("models/" + name + ".pkl", "rb") as f:
            self.V = pickle.load(f)

    def get_human_move(self, game: TTTGame):
        allowed_moves = [i + 1 for i in range(9) if game.state[i] == " "]
        human_move = None
        while not human_move:
            idx = int(
                input(
                    "Choose move for {}, from {} : ".format(game.player, allowed_moves)
                )
            )
            if any([i == idx for i in allowed_moves]):
                human_move = game.state[: idx - 1] + game.player + game.state[idx:]
        return human_move
