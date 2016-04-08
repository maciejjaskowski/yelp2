import boto3
from subprocess import call
import botocore


def upload_user_data(**kargs):

    s3 = boto3.resource('s3')
    s3.create_bucket(ACL='private', Bucket=kargs['client_token'])
    k = s3.Object(kargs['client_token'], 'config/run.sh')

    try:
        script = k.get()['Body'].read()
    except botocore.exceptions.ClientError as e:

        print("No run.sh not found. Uploading.")
        script = """#!/bin/bash
      echo "Install PIP"
      apt-get install -y python-pip
      apt-get install -y python-pandas
      mkdir /home/ubuntu/logs

      apt-get install -y ec2-api-tools

        """.format(**kargs)

        k.put(Body=script)
        call(["aws", "s3", "sync", "../analysis", "s3://" + kargs['exp_name'] + "/analysis"])
        call(["mkdir", "-p", "../" + kargs['exp_name']])
        call(["cp", "../analysis/sync.sh", "../" + kargs['exp_name']])
    else:
        print("run.sh found on S3. Reusing.")

    return script
