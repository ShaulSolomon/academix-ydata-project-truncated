{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled4.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMhI50IEKGUUSx9UMlBDuBU",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rgranit/academix-ydata-project/blob/master/code/s3_functions.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Tn90i8jl_w_Z",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from boto.s3.connection import S3Connection\n",
        "from boto.s3.key import Key\n",
        "from boto import s3\n",
        "import boto3, os, re, ssl\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "#Not sure what this does - but it was in the code.\n",
        "if (not os.environ.get('PYTHONHTTPSVERIFY', '') and\n",
        "getattr(ssl, '_create_unverified_context', None)):\n",
        "    ssl._create_default_https_context = ssl._create_unverified_context\n",
        "\n",
        "def get_dataframe_from_s3(AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY, key, bucket, file=\"data.csv\"):\n",
        "  conn = S3Connection(AWS_ACCESS_KEY, AWS_ACCESS_SECRET_KEY)\n",
        "  conn.auth_region_name = 'eu-west-1.amazonaws.com'\n",
        "  mybucket = conn.get_bucket(bucket)\n",
        "\n",
        "  # Retrieve Data\n",
        "  key = mybucket.get_key('key)\n",
        "  key.get_contents_to_filename(file)\n",
        "  df = pd.read_csv('data.csv')\n",
        "  return df\n",
        "\n",
        "def upload_to_s3(aws_access_key_id, aws_secret_access_key, file, bucket, key, callback=None, md5=None, reduced_redundancy=False, content_type=None):\n",
        "    \"\"\"\n",
        "    Uploads the given file to the AWS S3\n",
        "    bucket and key specified.\n",
        "    \n",
        "    callback is a function of the form:\n",
        "    \n",
        "    def callback(complete, total)\n",
        "    \n",
        "    The callback should accept two integer parameters,\n",
        "    the first representing the number of bytes that\n",
        "    have been successfully transmitted to S3 and the\n",
        "    second representing the size of the to be transmitted\n",
        "    object.\n",
        "    \n",
        "    Returns boolean indicating success/failure of upload.\n",
        "    \"\"\"\n",
        "    try:\n",
        "        size = os.fstat(file.fileno()).st_size\n",
        "    except:\n",
        "        # Not all file objects implement fileno(),\n",
        "        # so we fall back on this\n",
        "        file.seek(0, os.SEEK_END)\n",
        "        size = file.tell()\n",
        "    \n",
        "    conn = boto.connect_s3(aws_access_key_id, aws_secret_access_key)\n",
        "    bucket = conn.get_bucket(bucket, validate=True)\n",
        "    k = Key(bucket)\n",
        "    k.key = key\n",
        "    if content_type:\n",
        "        k.set_metadata('Content-Type', content_type)\n",
        "    sent = k.set_contents_from_file(file, cb=callback, md5=md5, reduced_redundancy=reduced_redundancy, rewind=True)\n",
        "    \n",
        "    # Rewind for later use\n",
        "    file.seek(0)\n",
        "    \n",
        "    if sent == size:\n",
        "        return True\n",
        "    return False\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}