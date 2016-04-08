import boto3
import base64
import time
from datetime import datetime
from gen_run import upload_user_data

ec2 = boto3.client('ec2')

instance_type = 'g2.2xlarge'


def _prices(az):
    next_token = ''
    a = []
    for i in range(1):
        res = ec2.describe_spot_price_history(StartTime=datetime(2016, 1, 1), InstanceTypes=[instance_type],
                                              AvailabilityZone=az, ProductDescriptions=['Linux/UNIX'], MaxResults=1000, NextToken=next_token)
        a = a + res['SpotPriceHistory']
        next_token = res['NextToken']

    return sorted(a, key=lambda s: s['Timestamp'])


def prices():
    return zip(_prices('us-east-1a'), _prices('us-east-1b'), _prices('us-east-1c'), _prices('us-east-1e'))



def provision(client_token, availability_zone, spot_price):

    user_data = """#!/bin/bash
      apt-get update
      echo "Install awscli"
      apt-get install -y awscli
      mkdir ~/.aws
      echo "[default]" >> ~/.aws/config
      echo "region = us-east-1" >> ~/.aws/config
      echo "output = json" >> ~/.aws/config
      echo "Download run.sh"
      aws s3 sync s3://{client_token}/config /home/ubuntu/
      echo "./run.sh"
      chmod a+x /home/ubuntu/run.sh
      /home/ubuntu/run.sh > /home/ubuntu/run_log.out 2> /home/ubuntu/run_log.err
    """.format(client_token=client_token)

    print(user_data)

    result = ec2.request_spot_instances(DryRun=False,
                                        ClientToken=client_token,
                                        SpotPrice=spot_price,
                                        InstanceCount=1,
                                        AvailabilityZoneGroup=availability_zone,
                                        Type='one-time',
                                        LaunchSpecification={
                                            'ImageId': 'ami-bdd2efd7',
                                            'KeyName': 'gpu-east',
                                            'InstanceType': instance_type,
                                            'Placement': {
                                                'AvailabilityZone': availability_zone
                                            },
                                            'BlockDeviceMappings': [{
                                                'DeviceName': '/dev/sda1',
                                                'Ebs': {
                                                    'VolumeSize': 25,
                                                    'DeleteOnTermination': True,
                                                    'VolumeType': 'standard',
                                                    'Encrypted': True
                                                }
                                            }],
                                            'IamInstanceProfile': {
                                                'Name': 's3'
                                            },
                                            'EbsOptimized': False,
                                            'Monitoring': {
                                                'Enabled': True
                                            },
                                            'UserData': base64.b64encode(user_data.encode("ascii")).decode('ascii'),
                                            'SecurityGroupIds': ['sg-ab1236d2'],

                                        })

    req_id = result['SpotInstanceRequests'][0]['SpotInstanceRequestId']

    print("")
    instance_description = None
    public_dns_name = ''

    while True:
        try:
            instance = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])['SpotInstanceRequests'][0]
            sys.stdout.write(str(datetime.now().time()) + " " + instance['Status']['Message'] + '\r')
            sys.stdout.flush()
            print(instance['Status']['Code'] )
            if instance['Status']['Code'] == 'instance-terminated-by-user':
                return {
                    'status': 'Killed',
                    'instance': None
                }

            if 'Fault' in instance.keys():
                return {
                    'status': instance['Status'],
                    'instance': instance
                }

            if 'InstanceId' in instance.keys():

                    instance_description = ec2.describe_instances(InstanceIds=[instance['InstanceId']])
                    public_dns_name = instance_description['Reservations'][0]['Instances'][0]['PublicDnsName']

            if public_dns_name != '':
                break
        except:
            print("sth went wrong. Let's wait a bit more.")
            pass

        time.sleep(1)

    return {
        'status': instance['Status'],
        'instance': instance,
        'instance_description': instance_description,
        'public_dns_name': public_dns_name
    }


def main(availability_zone, spot_price, exp_name, run_id):
    client_token = exp_name + '-' + run_id
    print("client_token: " + client_token)
    if not os.path.exists("../" + client_token):
        os.makedirs("../" + client_token)
        with open("../" + client_token + '/provision.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(' '.join(sys.argv))
    project_name = "yelp2"
    from subprocess import Popen, PIPE
    process = Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=PIPE)
    sha1, _ = process.communicate(str.encode("utf-8"))
    sha1 = sha1[:-1]

    user_script = upload_user_data(client_token=client_token, exp_name=exp_name, sha1=sha1, user_name="ubuntu",
                                   project_name=project_name)

    print("""
    project_name: {project_name}
    sha1: {sha1}

    availability_zone: {availability_zone}
    spot_price: {spot_price}

    exp_name: {exp_name}
    run_id: {run_id}
    client_token: {client_token}
    user_script:

    {user_script}
    """.format(project_name=project_name, sha1=sha1,
               client_token=client_token,
               exp_name=exp_name, run_id=run_id, user_script=user_script,
               spot_price=spot_price, availability_zone=availability_zone))

    instance = provision(client_token=client_token, availability_zone=availability_zone,
                         spot_price=spot_price)

    print(instance)
    # print("public_dns_name: ", instance['public_dns_name'])
    ec2_res = boto3.resource('ec2')
    i_id = instance['instance']['InstanceId']
    print(i_id)
    ec2_res.Instance(i_id).create_tags(Tags=[{'Key': 'Name', 'Value': client_token}])

    with open('instance.dns', 'w') as f:
        f.write(str(instance['public_dns_name']))

if __name__ == "__main__":
    import sys
    import getopt
    import os

    print("Running ")
    print(' '.join(sys.argv))

    optlist, args = getopt.getopt(sys.argv[1:], '', [
        'price=',
        'exp_name=',
        'run_id=',
        'stdin'
        ])

    d = {'availability_zone': 'us-east-1c'}
    stdin = False
    for o, a in optlist:
        if o in ("--price",):
            d['spot_price'] = a
        elif o in ("--stdin",):
            stdin = True
        elif o in ("--exp_name",):
            d['exp_name'] = a
        elif o in ("--run_id",):
            d['run_id'] = a
        else:
            assert False, "unhandled option {o} {a}".format(o=o, a=a)

    main(**d)


def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    plt.ion()
    fig, ax1 = plt.subplots()

    x, y = zip(*[(p['Timestamp'], p['SpotPrice']) for p in _prices('us-east-1c')])
    print(min(x))
    print(max(x))
    print(len(x))
    sx = pd.Series(x)
    delta = (sx - sx.shift(1))[1:]
    conc = pd.concat([delta, pd.Series(np.array(y, dtype=np.float32))], axis=1)
    conc.iloc[:, 1] = conc.iloc[:, 1].shift(1)

    df = pd.DataFrame(np.array(y, dtype=np.float32))
    bins = 25

    def calc(c):
        eligible = (pd.concat([df, df.shift(1)], axis=1) <= c)[1:]
        raised = eligible.apply(lambda x: not x.iloc[0] and x.iloc[1], axis=1)
        return len(np.where(raised)[0])

    def calc_time(c):
        return conc[conc.iloc[:, 1] < c].iloc[:, 0].sum().total_seconds() / 60 / 60

    lin = np.linspace(df.min()[0], df.max()[0], bins)

    print([calc_time(c) for c in lin])
    ax1.plot(lin, [calc_time(c) for c in lin])

    ax2 = ax1.twinx()

    ax2.plot(lin, [calc(c) for c in lin], color='r')
    ax2.set_ylabel('Liczba przerwan', color='r')

    fig2 = plt.figure()
    npx = np.array(x)
    npy = np.array(y)
    npx = np.apply_along_axis(lambda x: x[0].total_seconds() / 60 / 60, 1, np.reshape((npx - max(npx)), (len(npx), 1)))

    plt.plot(npx, npy)


