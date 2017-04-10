#coding=utf-8
__author__ = "yangl"
'''''
全局变量
'''
'''''
publickey为服务端的公钥
privatekey为自己客户端的私钥
PS：python的密钥都是PKCS1的
站点上的客户端私钥需要剔除
-----BEGIN RSA PRIVATE KEY-----
AND
-----END RSA PRIVATE KEY-----
并且将回车符剔除
'''

privatekey='''
-----BEGIN RSA PRIVATE KEY-----
MIICXQIBAAKBgQCCBvRpdck5L0ogutiLnTBLp7VdMZrHfeKN9CFQYQQ5/ocqNSkR
J/S6wUcWY6TlAwtfFRk/rBZn8S2dd/j6DOGZ8G4+WCtwBhp1v23WpcJYkMerJb3m
aTW9Z3XmA+MjPOsakMcxozmoO1CCOCXHTSICxgfKa3IejifAJwnyohXWbwIDAQAB
AoGAWQ0STzfP/E4a4peUvumErgvJ9m2Gp6Hbi4TrW/VVw7JCN/H4kjtfLJg6a2cL
A502KIR2qljdb4qJxxLnfblct+L7VaY+2rbrRFijus7bIlI1Sm0M6Y9NQ8Hic61H
UZNYYIIrB/hpEp4Tb4zzx8GbBx8I5VpJtSGB6sIpLbNI96ECQQDrxouMuXXbI6GW
HR27HddRu6d/Q43YAVoOeGKDlm32eGE6NDo1ZG+jCk5s+orJeluGRJPBdvHv7JCM
/1p+S7clAkEAjS5BAq0Jw0L26H/QeBShea6bzwuucHMtt0T7cBWYWqxgcFtg5y0y
5WXcAHG6jkVANSddy5eR8PBOxAUqTw2dAwJBAOnA87P2X3l+7wIUJdjQ8hvvb1XG
VQfV90InaoxJhQX6PXLmOtuaku/TFQQItbahH8KTlOYXFjCnmnyf4kkaqh0CQF0e
xByau9TCJ4+lNoDtwrA6/mQrZUyge+flJR+B7vLnvdh+PUVeJ7LtY5YbbZyHitlE
dPZjrAxKxPlAGu73oLcCQQCdILL/08tphRrKdHngVFqnQG++UMCTiTzwUWfnRgNr
4isH4ZUmTpsQqbDLWjAjOPmkl+R7a+0shr8bsZAFtcxe
-----END RSA PRIVATE KEY-----
'''

# https://ac.ppdai.com/oauth2/login?AppID=410a9843cf9444249b02e5cac2de9b95&ReturnUrl=http://ppdai.com
# http://www.ppdai.com/?code=17ef84b82a674697a20bbc8fde746a7b&state=
