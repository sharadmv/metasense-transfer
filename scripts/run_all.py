import click
from metasense import BOARD_CONFIGURATION as CONFIG

ROOT_DIR = 'results'
COMMAND = """set -ev;
mkdir -p {root_dir}/split/{server_name}/models
mkdir -p {root_dir}/split/{server_name}/subu
mkdir -p {root_dir}/split/{server_name}/linear
mkdir -p {root_dir}/split/{server_name}/nn-2
mkdir -p {root_dir}/split/{server_name}/nn-4
python scripts/plot_split.py Linear {root_dir}/split/{server_name}/linear {root_dir}/linear {root_dir}/split/{server_name}/models/*
python scripts/plot_split.py NN[2] {root_dir}/split/{server_name}/nn-2 {root_dir}/nn-2 {root_dir}/split/{server_name}/models/*
python scripts/plot_split.py NN[4] {root_dir}/split/{server_name}/nn-4 {root_dir}/nn-4 {root_dir}/split/{server_name}/models/*
python scripts/plot_split.py RF {root_dir}/split/{server_name}/subu {root_dir}/subu {root_dir}/split/{server_name}/models/*
"""

COMMAND_TEMPLATE = """tmux new-window -t {server_name}:{i} -n {name} 'bash -i'
tmux send-keys -t {server_name}:{i} 'venv; python scripts/train_split_model.py {root_dir}/split/{server_name}/models/{name} --round {round} --location {location} --board {board}; python scripts/generate_split.py {root_dir}/split/{server_name}/models/{name} --level1' Enter
"""

@click.command()
@click.argument('server_name')
@click.option('--root-dir', default=ROOT_DIR)
def generate_command(server_name, root_dir):
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
                                                  server_name=server_name,
                                                  root_dir=root_dir)
                commands.append(command)
                i += 1
    print(COMMAND.format(server_name=server_name, commands="\n".join(commands), root_dir=root_dir), end='')

if __name__ == "__main__":
    generate_command()
