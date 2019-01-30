from tictactoe import TTTGame, RLTicTacToeAgent
import click


@click.group()
def cli():
    pass


@click.command()
@click.option("--episodes", default=1000, help="Number of episodes")
@click.option("--alpha", default=0.2, help="Learning rate")
@click.option("--epsilon", default=0.1, help="Exploration rate")
@click.option("--save", default="primary_model", help="Model Save Path")
def train(episodes, alpha, epsilon, save):
    agent = RLTicTacToeAgent(TTTGame, epsilon, alpha)
    print("Learning with {} in memory games".format(episodes))
    agent.learn_game(episodes)
    file_name = agent.save_model(save)
    print("Model saved to {}".format(file_name))


@click.command()
@click.option("--model", default="primary_model", help="Model to use")
def play(model):
    agent = RLTicTacToeAgent(TTTGame)
    agent.load_model(model)
    agent.human_game()


cli.add_command(train)
cli.add_command(play)

if __name__ == "__main__":
    cli()
