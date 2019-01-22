import tqdm
import itertools
import s3fs
import time
import os
from path import Path
import socket
from argparse import ArgumentParser
from collections import defaultdict
from metasense import BOARD_CONFIGURATION
import boto3
from multiprocessing.pool import ThreadPool

BUCKET_NAME = "metasense-paper-results"

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('experiment')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--location', default=None, type=str)
    argparser.add_argument('--round', default=None, type=str)
    argparser.add_argument('--board', default=None, type=str)
    argparser.add_argument('--dim', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=20)
    argparser.add_argument('--hidden-size', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--load', default=None)
    argparser.add_argument('--num-iters', type=int, default=2000000)
    argparser.add_argument('--run', action='store_true')
    argparser.add_argument('--benchmark', action='store_true')
    return argparser.parse_args()

ec2 = boto3.client('ec2')
completed_requests = set()

NAME_MAP = {
    'elcajon': 'e',
    'shafter': 's',
    'donovan': 'd',
}
REVERSE_NAME_MAP = {
    'e': 'elcajon',
    's': 'shafter',
    'd': 'donovan',
}

def get_spot_status(request_id):
    while True:
        try:
            waiter = ec2.get_waiter('spot_instance_request_fulfilled')
            waiter.wait(SpotInstanceRequestIds=[request_id])
            response = ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id]
            )
            return response['SpotInstanceRequests'][0]['InstanceId']
        except:
            print('Waiting again...')

def get_instance_url(instance_id):
    response = ec2.describe_instances(
        InstanceIds=[instance_id]
    )
    return response['Reservations'][0]['Instances'][0]['PublicIpAddress']


def wait_on_ssh(instance_ip):
    connected = False
    while not connected:
        s = socket.socket()
        s.settimeout(4)
        try:
            s.connect((instance_ip, 22))
            connected = True
        except Exception:
            print('Failed to connect to %s...' % instance_ip)
            time.sleep(2)
        finally:
            s.close()

def run_remote(experiment, key_path=os.path.expanduser('~/.aws/metasense.pem')):
    print(experiment)
    experiment_name = "-".join("_".join((str(x[0]), NAME_MAP[x[1]], str(x[2]))) for x in experiment)
    print("Experiment name:", experiment_name)
    instance_type = 'm5.large'
    # ami = 'ami-005538161aa300c7b'
    # ami = 'ami-094b2cdcc1b66470e'
    ami = 'ami-09fc54c1e28d2eae2'
    spot_price = '0.5'
    round = ",".join([str(x[0]) for x in experiment])
    location = ",".join([str(x[1]) for x in experiment])
    board = ",".join([str(x[2]) for x in experiment])
    response = ec2.request_spot_instances(
        AvailabilityZoneGroup='us-west-2',
        LaunchSpecification=dict(
            SecurityGroups=['metasense-default'],
            ImageId=ami,
            InstanceType=instance_type,
            KeyName='metasense',
        ),
        SpotPrice=spot_price,

    )
    if len(response['SpotInstanceRequests']) > 0:
        for request in response['SpotInstanceRequests']:
            request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            if request_id not in completed_requests:
                completed_requests.add(request_id)
                break
    else:
        request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    instance_id = get_spot_status(request_id)
    print('Spun up instance:', instance_id)
    print('Setting instance properties')
    ec2.modify_instance_attribute(
        InstanceId=instance_id,
        BlockDeviceMappings=[{
            'DeviceName': '/dev/sda1',
            'Ebs': {
                'DeleteOnTermination': True
            }
        }]
    )
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            { 'Key': "Name", 'Value': experiment_name }
        ]
    )
    url = get_instance_url(instance_id)
    print('Spot instance url:', url)
    wait_on_ssh(url)
    os.system("ssh -ostricthostkeychecking=no -i {key_path} ubuntu@{url} 'tmux new-session -d -s {experiment_name} \"~/venv/bin/python scripts/train_split_model.py {experiment_type} {experiment_name} --hidden-size {hidden_size} --dim {dim} --num-iters {num_iters} --round {round} --location {location} --board {board} --lr {lr} --batch-size {batch_size}; tmux detach\"'".format(
        key_path=key_path,
        experiment_type=args.experiment,
        experiment_name=experiment_name,
        url=url,
        round=round,
        location=location,
        board=board,
        hidden_size=args.hidden_size,
        dim=args.dim,
        num_iters=args.num_iters,
        lr=args.lr,
        batch_size=args.batch_size
    ))

def get_location_experiment():
    for round in BOARD_CONFIGURATION:
        locations = BOARD_CONFIGURATION[round].keys()
        location_triples = {}
        for location in locations:
            for board in BOARD_CONFIGURATION[round][location]:
                if location not in location_triples:
                    location_triples[location] = set()
                location_triples[location].add((round, location, board))
        yield from [frozenset((a, b, c)) for a in location_triples['shafter'] for b in location_triples['elcajon'] for c in location_triples['donovan']]

def get_seasonal_experiment():
    # round_triples = {}
    # for round in BOARD_CONFIGURATION:
        # locations = BOARD_CONFIGURATION[round].keys()
        # for location in locations:
            # for board in BOARD_CONFIGURATION[round][location]:
                # if round not in round_triples:
                    # round_triples[round] = set()
                # round_triples[round].add((round, location, board))
    # yield from [
        # frozenset((a, b, c))
        # for a in round_triples[2]
        # for b in round_triples[3]
        # for c in round_triples[4]
        # if a[1] == b[1] == c[1]
    # ]
    triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                triples.add((round, location, board))
    combos = list(itertools.combinations(triples, 3))
    yield from [
        frozenset((a, b, c)) for a, b, c in combos
        if ((a[0] != b[0] and a[0] != c[0] and b[0] != c[0])
            and (a[1] == b[1] and a[1] == c[1] and b[1] == c[1])
            and (a[2] != b[2] and a[2] != c[2] and b[2] != c[2]))
    ]

def get_locationseasonal_experiment():
    triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                triples.add((round, location, board))
    combos = list(itertools.combinations(triples, 3))
    yield from [
        frozenset((a, b, c)) for a, b, c in combos
        if ((a[0] != b[0] and a[0] != c[0] and b[0] != c[0])
            and (a[1] != b[1] and a[1] != c[1] and b[1] != c[1])
            and (a[2] != b[2] and a[2] != c[2] and b[2] != c[2]))
    ]

def get_seasonal_size(n):
    triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                triples.add((round, location, board))
    combos = itertools.combinations(triples, n)
    for combo in combos:
        rounds = [c[0] for c in combo]
        locations = [c[1] for c in combo]
        round_set = set(rounds)
        if not all([l == locations[0] for l in locations]):
            continue
        if not len(round_set) == 3:
            continue
        num_rounds = int(n / 3)
        if [rounds.count(l) for l in round_set] != [num_rounds] * 3:
            continue
        yield frozenset(combo)

def get_location_size(n, level=1):
    triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                triples.add((round, location, board))
    combos = itertools.combinations(triples, n * level)
    for combo in tqdm.tqdm(combos):
        rounds = [c[0] for c in combo]
        round_set = set(rounds)
        locations = [c[1] for c in combo]
        location_set = set(locations)
        boards = [c[2] for c in combo]
        board_set = set([c[2] for c in combo])
        board_counts = [boards.count(b) for b in board_set]
        if len(board_set) != n:
            continue
        if board_counts != [level] * len(board_set):
            continue
        if not len(location_set) == 3:
            continue
        if not len(round_set) == level:
            continue
        num_locations = int(n * level / 3)
        if [locations.count(l) for l in location_set] != [num_locations] * 3:
            continue
        yield frozenset(combo)

def get_seasonal_size(n, level=1):
    triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                triples.add((round, location, board))
    combos = itertools.combinations(triples, n * level)
    for combo in tqdm.tqdm(combos):
        rounds = [c[0] for c in combo]
        round_set = set(rounds)
        locations = [c[1] for c in combo]
        location_set = set(locations)
        boards = [c[2] for c in combo]
        board_set = set([c[2] for c in combo])
        board_counts = [boards.count(b) for b in board_set]
        if len(board_set) != n:
            continue
        if board_counts != [level] * len(board_set):
            continue
        if not len(location_set) == level:
            continue
        if not len(round_set) == 3:
            continue
        num_rounds = int(n * level / 3)
        if [rounds.count(l) for l in round_set] != [num_rounds] * 3:
            continue
        yield frozenset(combo)

if __name__ == "__main__":
    args = parse_args()
    experiments = set()
    board_map = defaultdict(list)
    all_experiments = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                all_experiments.add((round, location, board))
    all_experiments = frozenset(all_experiments)
    # for experiment in get_location_experiment():
    # for experiment in get_locationseasonal_experiment():
    for experiment in get_location_size(6, level=1):
        print(experiment)
        experiments.add(all_experiments - experiment)
    # for round in BOARD_CONFIGURATION:
        # for location in BOARD_CONFIGURATION[round]:
            # for board in BOARD_CONFIGURATION[round][location]:
                # experiments.add(frozenset(((round, location, board),)))
                # board_map[board].append((round, location))
    # for board in board_map:
        # triples = set((a, b, board) for a, b in board_map[board])
        # for triple in triples:
            # experiments.add(frozenset(triples - {triple}))
    # for round in BOARD_CONFIGURATION:
        # for location in BOARD_CONFIGURATION[round]:
            # for board in BOARD_CONFIGURATION[round][location]:
                # experiments.add(frozenset(((round, location, board),)))
                # board_map[board].append((round, location))
    # for board in board_map:
        # triples = set((a, b, board) for a, b in board_map[board])
        # for triple in triples:
            # experiments.add(frozenset(triples - {triple}))
    fs = s3fs.S3FileSystem(anon=False)
    out_dir = Path(BUCKET_NAME) / args.experiment
    print(out_dir)
    def process(x):
        a, b, c = x.split("_")
        return (int(a), REVERSE_NAME_MAP[b], int(c))
    try:
        existing = set(frozenset(map(process, frozenset(x.split('/')[-1].split("-")))) for x in fs.ls(out_dir))
    except:
        existing = set()
    print("Number of experiments:", len(experiments))
    print("Number of existing:", len(existing))
    # print(experiments)
    if args.run:
        # experiments -= existing
        print(experiments)
        pool = ThreadPool(10)
        pool.map(run_remote, list(experiments))
    if args.benchmark:
        existing = [x.split('/')[-1] for x in fs.ls(out_dir)]
        commands = []
        def process(x):
            a, b, c = x
            return (str(a), NAME_MAP[b], str(c))
        for experiment in experiments:
            experiment = list(experiment)
            if len(experiment) == 2:
                es = ["-".join("_".join(process(x)) for x in e) for e in [experiment, reversed(experiment)]]
            else:
                es = ["-".join("_".join(process(x)) for x in experiment)]
            for e in existing:
                commands.append("python scripts/generate_split.py %s %s --level1" % (args.experiment, e))
        import ipdb; ipdb.set_trace()
        pool = ThreadPool(10)
        pool.map(os.system, commands)
