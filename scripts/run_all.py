import click
from metasense import BOARD_CONFIGURATION as CONFIG

COMMAND = """set -ev;
mkdir -p results/{server_name}/models
mkdir -p results/{server_name}/subu
mkdir -p results/{server_name}/linear
mkdir -p results/{server_name}/nn-2
mkdir -p results/{server_name}/nn-4
tmux new-session -s {server_name} -n start -d
{commands}
python scripts/plot_split.py results/{server_name}/linear results2/linear results/{server_name}/models/*
python scripts/plot_split.py results/{server_name}/nn-2 results2/nn-2 results/{server_name}/models/*
python scripts/plot_split.py results/{server_name}/nn-4 results2/nn-4 results/{server_name}/models/*
python scripts/plot_split.py results/{server_name}/subu results2/subu results/{server_name}/models/*
"""

COMMAND_TEMPLATE = """tmux new-window -t {server_name}:{i} -n {name} 'bash -i'
tmux send-keys -t {server_name}:{i} 'venv; python scripts/train_split_model.py {server_name}/models/{name} --round {round} --location {location} --board {board}; python scripts/generate_split.py {server_name}/models/{name} --level1' Enter
"""

@click.command()
@click.argument('server_name')
def generate_command(server_name):
    commands = []
    i = 2
    for round in CONFIG:
        for location in CONFIG[round]:
            for board in CONFIG[round][location]:
                name = "{}-{}-{}".format(round, location, board)
                command = COMMAND_TEMPLATE.format(i=i, name=name,
                                                  round=round,
                                                  location=location,
                                                  board=board,
                                                  server_name=server_name)
                commands.append(command)
                i += 1
    print(COMMAND.format(server_name=server_name, commands="\n".join(commands)), end='')

if __name__ == "__main__":
    generate_command()
