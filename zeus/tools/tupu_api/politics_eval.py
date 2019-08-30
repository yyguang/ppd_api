#encoding:utf-8
# Author fujunhao <fujunhao@ishumei.com>
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
"""
1. Install Dependencies
sudo pip install rsa requests base64 json

2. Interface
tupu_client = TUPU(secret_id, private_key_path, url)
parameter: "url" default is "http://api.open.tuputech.com/v3/recognition/"

3. Example

from tupu_api import TUPU
tupu = TUPU(secret_id="xxxxxxxxxxxxxxxxxx",
            private_key_path="./rsa_private_key.pem")
# url
images = ["http://example.com/001.jpg", "http://example.com/002.jpg"]
result = tupu.api(images=images, is_url=True)

# image file
images = ["/home/user/001.jpg", "/home/user/002.jpg"]
result = tupu.api(images=images, is_url=False)

# zip file
images = ["/home/user/001.zip", "/home/user/002.zip"]
result = tupu.api(images=images, is_url=False)
"""
import traceback
import os
import random
import datetime
import rsa
import requests
import base64
import json
import sys

TUPU_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDyZneSY2eGnhKrArxaT6zswVH9
/EKz+CLD+38kJigWj5UaRB6dDUK9BR6YIv0M9vVQZED2650tVhS3BeX04vEFhThn
NrJguVPidufFpEh3AgdYDzOQxi06AN+CGzOXPaigTurBxZDIbdU+zmtr6a8bIBBj
WQ4v2JR/BA6gVHV5TwIDAQAB
-----END PUBLIC KEY-----
"""
# taskId为图普定义的暴恐2.0接口标识
taskId='59101e2540f7575536ab3144'

class TUPU:
    def __init__(self, secret_id, private_key_path, url='http://api.open.tuputech.com/v3/recognition/'):
        self.__url = url + ('' if url.endswith('/') else '/') + secret_id
        self.__secret_id = secret_id
        # get private key
        with open(private_key_path) as private_key_file:
            self.__private_key = rsa.PrivateKey.load_pkcs1(private_key_file.read())
        # get tupu public key
        self.__public_key = rsa.PublicKey.load_pkcs1_openssl_pem(TUPU_PUBLIC_KEY)

    def __sign(self):
        """get the signature"""
        self.__timestamp = datetime.datetime.now()
        self.__nonce = random.randint(1 << 4, 1 << 32)
        sign_string = "%s,%s,%s" % (self.__secret_id, self.__timestamp, self.__nonce)
        self.__signature = base64.b64encode(rsa.sign(sign_string.encode("utf-8"), self.__private_key, 'SHA-256'))

    def __verify(self, signature, verify_string):
        """verify the signature"""
        try:
            rsa.verify(verify_string.encode("utf-8"), base64.b64decode(signature), self.__public_key)
            return "Success"
        except rsa.pkcs1.VerificationError:
          print("Verification Failed")
        return "Failed"

    def api(self, images, is_url=False):
        if not isinstance(images, list):
            raise Exception('[ArgsError] images is a list')
        self.__sign()
        request_data = {
            "timestamp": self.__timestamp,
            "nonce": self.__nonce,
            "signature": self.__signature
        }
        response = None
        if is_url:
            request_data["image"] = images
            response = requests.post(self.__url, data=request_data)
        else:
            multiple_files = []
            for image_file in images:
                if not os.path.isfile(image_file):
                    print('[SKIP FILE] No such file "%s"' % image_file)
                    continue
                multiple_files.append(('image', (image_file, open(image_file, 'rb'), 'application/*')))
            response = requests.post(self.__url, data=request_data,
            files=multiple_files, timeout=10)
        response_json = json.loads(response.text)
        #print(response_json)
        if not "error" in response_json:
            response_json['verify_result'] = self.__verify(response_json['signature'], response_json['json'])
            response_json['json'] = json.loads(response_json['json'])
        return response_json

def call_tupu(images):
  for j in xrange(3):
    try:
      result = tupu.api(images=images, is_url=False)
      break
    except Exception as e:
      print(traceback.format_exc())

  if j == 2:
    for image in images:
      print("failed " + image)
    sys.stdout.flush()
    return False
  else:
    #images = []
    #print result
    fileList = result["json"][taskId]["fileList"]
    #print "count = " + str(count)
    tmp_count = len(fileList)
    for i in xrange(tmp_count):
      file_name = fileList[i]["name"]
      label = fileList[i]["label"]
      review = fileList[i]["review"]
      faceid = fileList[i]["faceId"]
      similarity = fileList[i]["similarity"]
      print("%s\t%s\t%s\t%s\t%s" %(file_name, label, review, faceid, similarity))
    sys.stdout.flush()
    return True


if __name__ == "__main__":
    if len(sys.argv) != 4:
      print("usage : tupu_api file_list_file log_file_path")
      sys.exit(-1)
    tupu = TUPU(secret_id='589994aa189cf2926489dc39',
                        private_key_path='./private_key')
    ## image file
    with open(sys.argv[1], "r") as f:
      images = []
      count = 1
      for line in f:
        arr = line.strip().split('\t')
        file_name = arr[0]
        face_num = arr[1]
        faceid = arr[2]
        dis = arr[3]
        images.append(file_name)
        if count % 20 == 0:
          if call_tupu(images):
            images = []
        count +=1
    if len(images) > 0:
      call_tupu(images)
