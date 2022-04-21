import requests
import json
from flask import Flask, request, jsonify


#1 feature_extraction
# url = 'http://127.0.0.1:9090/feature_extraction/face'

# features = []
# for i in range(1,3):
#     data = {
#         "pictureId": str(i),
#         "screenshot": "samples/{}.png".format(i)
#     }
#     data = json.dumps(data)
#     r = requests.post(url, data)
#     r = json.loads(r.text)
#     print(r)
#     features.append(r["message"]["feature"])


# #2 feature_comparison
# url = 'http://127.0.0.1:9090/feature_comparison'
# feature = "0.28124192 -0.42538434 -0.6094999 -1.3636254 -0.27577603 1.1586905 -0.2517332 -1.487151 -0.24273846 0.48348135 0.7650865 0.6913335 0.60066384 0.2852897 -0.6021562 0.9274651 -2.285186 -0.8539836 -0.82722294 1.3440092 0.873991 -0.516291 1.5447564 1.4096831 1.1506662 -1.3982491 -0.4268116 0.4030607 0.6917173 1.0405412 -0.609 -0.35088196 1.5529419 0.6029823 1.3387462 0.91696084 -1.4183003 1.1905066 0.95020604 0.24502541 -1.0874869 1.3473046 -2.0340273 1.935839 0.9281686 0.97411686 0.5816157 0.46800897 0.6560615 0.7656222 -0.7821058 0.9068303 -0.50750124 0.28792185 0.6291372 0.06687686 -1.1058362 -0.8121676 -0.2411867 -0.10020561 -1.0786245 -0.13365172 0.035642035 1.214334 1.5637121 -1.5397819 0.2998848 1.6237223 -0.9643154 1.0096332 0.34335664 -0.27582556 -1.2992971 -0.052337043 0.104748785 1.2276645 1.1024338 -0.08321311 0.5033081 0.43911755 -1.1242994 -0.3291359 0.86494946 1.2072012 1.6750331 0.53686905 -1.893181 0.33091766 1.0335883 1.4270673 -0.21802181 0.3510457 0.82594675 -1.1796744 0.9088737 -0.48277387 -0.404034 -1.1645764 -0.6232488 -0.7586862 0.8961879 0.10821165 -0.72501636 -0.1790734 -1.7198782 -0.26050258 1.2122436 1.2408254 -0.23190331 1.8821028 0.3992072 0.8668422 -0.95901674 -0.524424 0.43420985 -2.285052 -0.75780797 0.19882925 -1.035208 0.102442786 0.32742006 -0.4992252 0.010443881 0.86671996 0.20427589 -0.33484292 0.42176288 0.41365898 0.6804812 -2.429829 -0.16670218 -2.4599893 -1.26365 0.9093092 1.2014802 1.2818084 0.25265 -0.116966955 0.61628544 -0.27717558 -2.1062179 -0.53887063 -0.0010707221 -0.92813885 -0.9578658 0.23229316 -0.14606656 -1.1685414 0.3855742 -1.0455964 1.3014672 -0.9563203 2.4686694 0.5123538 1.3662544 0.8429347 -0.03019047 1.0861638 0.37619776 -0.9283423 0.25286245 0.787608 -0.30072102 -0.81157666 1.727217 -3.1011214 0.50629634 1.0879308 -0.37621596 1.7873753 0.49167228 0.9304904 1.7063478 1.5662941 0.65928644 0.83467025 -0.6345945 -0.4537796 0.45342377 -0.44571817 0.9933613 -0.17422462 -0.57466143 0.6162587 0.4836498 1.246029 1.3337506 -0.7213458 0.38365304 "

# data = {
#     "feature":features,
#     "isMode":1,
#     "baseFeatures":[
#       {
#        "pictureId":str(i),
#        "feature":features[i]
#       } for i in range(len(features))
#     ]
# }
# data = json.dumps(data)
# r = requests.post(url, data)
# r = json.loads(r.text)
# print(r)


# #3 /target_locate/face
# url = "http://127.0.0.1:8080/target_locate/face"

# data = {
#     "pictureId": "1001",
#     "screenshot": "samples/testperson4.jpg"
# }
# data = json.dumps(data)
# r = requests.post(url, data)
# r = json.loads(r.text)
# print(r)

# 4 /target_locate/vehicle
# url = "http://127.0.0.1:9090/target_locate/vehicle"

# data = {
#     "pictureId": "1001",
#     "screenshot": "/upload/xmyc/picture/1452670397353959425/20211125130341774.jpg"
# }

# data = json.dumps(data)
# r = requests.post(url, data)
# r = json.loads(r.text)
# print(r)


# # 5 /get_platemess
# url = "http://127.0.0.1:9090/get_platemess"

# data = {
#     "pictureId": "1001",
#     "screenshot": "samples/testcolor1.jpg"
# }

# data = json.dumps(data)
# r = requests.post(url, data)
# r = json.loads(r.text)
# print(r)


# # 6 /get_vehicletype
# url = "http://127.0.0.1:9090/get_vehicletype"

# data = {
#     "pictureId": "1001",
#     "screenshot": "samples/testcar2.jpg"
# }

# data = json.dumps(data)
# r = requests.post(url, data)
# r = json.loads(r.text)
# print(r)



# # 7 /identify_monitortime
url = "http://127.0.0.1:9090/identify_monitortime"

data = {
    "pictureId": "1001",
    "screenshot": "samples/testocr_200.png"
}

data = json.dumps(data)
r = requests.post(url, data)
r = json.loads(r.text)
print(r)


# # 8 /identify_monitorloc
url = "http://127.0.0.1:9090/identify_monitorloc"

data = {
    "pictureId": "1001",
    "screenshot": "samples/testocr_200.png"
}

data = json.dumps(data)
r = requests.post(url, data)
r = json.loads(r.text)
print(r)


# 9 /target_locate/object
# url = "http://127.0.0.1:9090/target_locate/object"

# data = {
#     "pictureId": "1001",
#     "screenshot": "/upload/xmyc/picture/2021-12/38de6596-1c14-4053-8cf4-f76620e33e8f.png"
# }

# data = json.dumps(data)
# r = requests.post(url, data)
# r = json.loads(r.text)
# print(r)
