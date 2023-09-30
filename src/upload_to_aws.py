from boto.s3.key import Key
from boto.s3.connection import S3Connection
import os
import json

def init_s3(bucket_name='groot-evaluation', access_keys='aws_access_keys_user009.json'):
  ACCESS_KEY, SECRET_KEY = json.load(open(access_keys, 'r'))
  conn = S3Connection(ACCESS_KEY, SECRET_KEY)
  b = conn.get_bucket(bucket_name)
  return b

def upload_file(src, tgt, bucket, force=False):
  if os.path.exists(src):
    k = Key(bucket)
    k.key = tgt
    if not k.exists() or force:
      k.set_contents_from_filename(src)
    k.make_public()
  else:
    tqdm.write('Not Found: {}'.format(src))

## initialize bucket
bucket = init_s3('groot-evaluation')

## write file
upload_file('a/b/file.mp4', 'a/b/file.mp4', bucket)
## the uploaded file can now be found at https://groot-evaluation.s3.amazonaws.com/a/b/file.mp4