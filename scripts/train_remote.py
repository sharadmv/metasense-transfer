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
            print('Failed to connect...')
            time.sleep(2)
        finally:
            s.close()

def run_remote(experiment, key_path=os.path.expanduser('~/.aws/metasense.pem')):
    print(experiment)
    experiment_name = "-".join("_".join(map(str, x)) for x in experiment)
    print("Experiment name:", experiment_name)
    instance_type = 'm5.large'
    # ami = 'ami-019728aa43c61ac9c'
    ami = 'ami-0aff95c47247db35e'
    spot_price = '0.3'
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

if __name__ == "__main__":
    args = parse_args()
    experiments = set()
    all_triples = set()
    # board_map = defaultdict(list)
    LOCATIONS = {'elcajon', 'shafter', 'donovan'}
    location_map = defaultdict(set)
    all_triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                location_map[location].add((round, location, board))
                all_triples.add((round, location, board))
    for location, es in location_map.items():
        experiments.add(frozenset(all_triples - es))
    # for location in LOCATIONS:
        # experiment = set()
        # for round in BOARD_CONFIGURATION:
            # for board in BOARD_CONFIGURATION[round][location]:
                # experiment.add((round, location, board))
                # all_triples.add((round, location, board))
        # experiments.add(frozenset(experiment))
    # for round in BOARD_CONFIGURATION:
        # for location in BOARD_CONFIGURATION[round]:
            # for board in BOARD_CONFIGURATION[round][location]:
                # experiments.add(frozenset(((round, location, board),)))
                # board_map[board].append((round, location))
    # for board in board_map:
        # triples = set((a, b, board) for a, b in board_map[board])
        # experiments.add(frozenset(all_triples - triples))
    fs = s3fs.S3FileSystem(anon=False)
    out_dir = Path(BUCKET_NAME) / args.experiment
    def process(x):
        a, b, c = x.split("_")
        return (int(a), b, int(c))
    try:
        existing = set(frozenset(map(process, frozenset(x.split('/')[-1].split("-")))) for x in fs.ls(out_dir))
        existing_map = {frozenset(map(process, frozenset(x.split('/')[-1].split("-")))):x for x in fs.ls(out_dir)}
    except:
        existing = set()
    print("Number of experiments:", len(experiments))
    print("Number of existing:", len(existing))
    if args.run:
        experiments -= existing
        print(experiments)
        pool = ThreadPool(10)
        pool.map(run_remote, list(experiments))
    if args.benchmark:
        commands = []
        for experiment in experiments:
            experiment = list(experiment)
            for ex in existing:
                if ex == frozenset(experiment):
                    e = Path(existing_map[ex]).basename()
            commands.append("python scripts/generate_split.py %s %s --level1" % (args.experiment, e))
        pool = ThreadPool(10)
        pool.map(os.system, commands)
