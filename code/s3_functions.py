# -*- coding: utf-8 -*-

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3, os, re, ssl
import pandas as pd
import numpy as np

#Not sure what this does - but it was in the code.
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def get_dataframe_from_s3(AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY, key, bucket, file="data.csv"):
  conn = S3Connection(AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY)
  conn.auth_region_name = 'eu-west-1.amazonaws.com'
  mybucket = conn.get_bucket(bucket)

  # Retrieve Data
  key = mybucket.get_key(key)
  key.get_contents_to_filename(file)
  df = pd.read_csv('data.csv')
  return df

def upload_to_s3(aws_access_key_id, aws_secret_access_key, file, bucket, key, callback=None, md5=None, reduced_redundancy=False, content_type=None):
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
    sent = k.set_contents_from_file(file, cb=callback, md5=md5, reduced_redundancy=reduced_redundancy, rewind=True)
    
    # Rewind for later use
    file.seek(0)
    
    if sent == size:
        return True
    return False