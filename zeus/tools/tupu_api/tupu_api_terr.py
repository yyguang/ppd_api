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
import logging

TUPU_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDyZneSY2eGnhKrArxaT6zswVH9
/EKz+CLD+38kJigWj5UaRB6dDUK9BR6YIv0M9vVQZED2650tVhS3BeX04vEFhThn
NrJguVPidufFpEh3AgdYDzOQxi06AN+CGzOXPaigTurBxZDIbdU+zmtr6a8bIBBj
WQ4v2JR/BA6gVHV5TwIDAQAB
-----END PUBLIC KEY-----
"""
logging.basicConfig(level=logging.DEBUG,
                    filename=sys.argv[2],
                    filemodel='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# taskId为图普定义的暴恐2.0接口标识
taskId='5808841f5e1778ef49219a99'
label_dict={"baoluanchangjing":0,"guoqiguohui":1,"junzhuang":2,"kongbuzuzhi":3,"qiangzhidaoju":4,"xuexingchangjing":5,"zhengchang":6}
label_dict2={"zhengchang":0,"tedingrenwu":1,"teshuzhuozhuang":2,"teshufuhao":3,"wuqi":4,
             "guojiadiqu":5,"xuexingchangjiang":6,"baoluanchangjing":7,"zhanzhengchangjing":8}

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
          logging.error ("Verification Failed")
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
                    loggin.error ('[SKIP FILE] No such file "%s"' % image_file)
                    continue
                multiple_files.append(('image', (image_file, open(image_file, 'rb'), 'application/*')))
            response = requests.post(self.__url, data=request_data,
            files=multiple_files, timeout=10)
        response_json = json.loads(response.text)
        logging.info(response_json)
        if not "error" in response_json:
            response_json['verify_result'] = self.__verify(response_json['signature'], response_json['json'])
            response_json['json'] = json.loads(response_json['json'])
        return response_json

if __name__ == "__main__":
    if len(sys.argv) != 4:
      print("usage : tupu_api file_list_file log_file_path")
      sys.exit(-1)
    print(len(label_dict2))
    label_dict2_sorted = sorted(label_dict2.values())
    new_dict = {v: k for k, v in label_dict2.items()}
    for v in label_dict2_sorted:
      print('%d %s' % (v, new_dict[v]))
    tupu = TUPU(secret_id='58acf776b6f8c710f1fef257',
                        private_key_path='./private_key')
    ## image file
    with open(sys.argv[1], "r") as f:
      count=1
      images = []
      for line in f:
        arr = line.strip().split()
        file_name = arr[0]
        images.append(file_name)
        if count % 20 == 0:
          j=0
          for j in range(3):
            try:
              result = tupu.api(images = images, is_url=False)
              break
            except Exception as e:
              logging.error(traceback.format_exc())

          if j == 2:
            for image in images:
              logging.error("failed " + image)
          else:
            images = []
            logging.info(result)
            fileList = result["json"][taskId]["fileList"]
            logging.info("count = " + str(count))
            tmp_count = len(fileList)
            for i in range(tmp_count):
              file_name = fileList[i]["name"]
              if fileList[i]["review"]:
                is_review = 1
              else:
                is_review = 0
              rate = fileList[i]["rate"]
              label = fileList[i]["label"]
              if label == -1:
                continue
              else:
                true_label = label_dict[arr[1]]
                rates = [-1] * len(label_dict2)
                rates[label] = rate
                print("%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d" % (
                file_name, label, rates[0], rates[1], rates[2], rates[3], rates[4], rates[5], rates[6], rates[7],rates[8], true_label))
          sys.stdout.flush()
        count += 1

      if len(images) > 0:
        #last
        for j in range(3):
          try:
            result = tupu.api(images = images, is_url = False)
            break
          except Exception as e:
            logging.error(traceback.format_exc())

        if j == 2:
          for image in images:
            logging.error("failed " + image)
        else:
          images = []
          logging.info(result)
          fileList = result["json"][taskId]["fileList"]
          logging.info("count = " + str(count))
          tmp_count = len(fileList)
          for i in range(tmp_count):
            file_name = fileList[i]["name"]
            if fileList[i]["review"]:
              is_review = 1
            else:
              is_review = 0
            rate = fileList[i]["rate"]
            label = fileList[i]["label"]
            if label == -1:
              continue
            else:
              true_label = label_dict[arr[1]]
              rates = [-1]*len(label_dict2)
              rates[label] = rate
              print("%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d" 
              %(file_name, label, rates[0],rates[1],rates[2],rates[3],rates[4],rates[5],rates[6],rates[7],rates[8],true_label))
          sys.stdout.flush()
