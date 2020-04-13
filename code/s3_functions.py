# -*- coding: utf-8 -*-

import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3
import boto
import re
import ssl
import pandas as pd
import numpy as np
from configparser import ConfigParser

### fetch the credentials ###
creds_path = "credentials.ini"
config_parser = ConfigParser()
config_parser.read(creds_path)
s3_dict = config_parser["S3"]
AWS_ACCESS_KEY = s3_dict['AWS_ACCESS_KEY']
AWS_ACCESS_SECRET_KEY = s3_dict['AWS_ACCESS_SECRET_KEY']
BUCKET = s3_dict['BUCKET']
#####


def get_creds():
        return s3_dict


def list_files_in_bucket(AWS_ACCESS_KEY=AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY=AWS_ACCESS_SECRET_KEY, bucket=BUCKET):
        """
        returns a list of file in a bucket
        """
        conn = S3Connection(AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY)
        conn.auth_region_name = 'eu-west-1.amazonaws.com'
        mybucket = conn.get_bucket(bucket)
        return [i for i in mybucket.list()]


def get_dataframe_from_s3(AWS_ACCESS_KEY=AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY=AWS_ACCESS_SECRET_KEY, bucket=BUCKET, file="data.csv"):
  conn = S3Connection(AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY)
  conn.auth_region_name = 'eu-west-1.amazonaws.com'
  mybucket = conn.get_bucket(bucket)

  # Retrieve Data
  key = mybucket.get_key(file)
  key.get_contents_to_filename(file)
  df = pd.read_csv(file)
  return df


def upload_to_s3(file, key, AWS_ACCESS_KEY=AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY=AWS_ACCESS_SECRET_KEY, bucket=BUCKET,  callback=None, md5=None, reduced_redundancy=False, content_type=None):
    """
    Uploads the given file to the AWS S3
    bucket and key specified.
    
    callback is a function of the form:
    
    def callback(complete, total)
    
    The callback should accept two integer parameters,
    the first representing the number of bytes that
    have been successfully transmitted to S3 and the
    second representing the size of the to be transmitted
    object.
    
    Returns boolean indicating success/failure of upload.
    """
    try:
        size = os.fstat(file.fileno()).st_size
    except:
        # Not all file objects implement fileno(),
        # so we fall back on this
        file.seek(0, os.SEEK_END)
        size = file.tell()

    conn = boto.connect_s3(aws_access_key_id, aws_secret_access_key)
    bucket = conn.get_bucket(bucket, validate=True)
    k = Key(bucket)
    k.key = key
    if content_type:
        k.set_metadata('Content-Type', content_type)
    sent = k.set_contents_from_file(
        file, cb=callback, md5=md5, reduced_redundancy=reduced_redundancy, rewind=True)

    # Rewind for later use
    file.seek(0)

    if sent == size:
        return True
    return False


def solve_mac_issue():
        #code used to solve some issue with ssl in mac
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
                ssl._create_default_https_context = ssl._create_unverified_context
