import requests
import json
import os
import cv2


# data = {"pictureId":20211201101245396,"screenshot":"/upload/xmyc/picture/2021-12/38de6596-1c14-4053-8cf4-f76620e33e8f.png"}
# data = json.dumps(data)
# url = 'http://0.0.0.0:9090/target_locate/object'
# r = requests.post(url, data)
# print(r.text)
# exit()

url = 'http://0.0.0.0:9090/feature_extraction/face'

features = []

# root_path = '/upload/xmyc/picture/2021-11/'
# root_path = '/home/nqiv_admin/ycnqiv/test1201/'
root_path = ''

# fpaths = os.listdir(root_path)
# fpaths.sort()
# fpaths = fpaths[:1000]
# print(fpaths)

fpaths = [
    '/upload/xmyc/picture/2021-11/76e6e40b-0a4f-4443-920e-3348efca6e83.png',
    '/upload/xmyc/picture/2021-11/65f1762b-78e2-400e-a6f8-57366dfb3e56.jpg',
    '/upload/xmyc/picture/2021-11/b28fdd7e-3066-4beb-a24b-5227748a0870.png',
    '/upload/xmyc/picture/2021-11/67e766a6-a976-4489-a954-8a0fc7706fd2.png',
    '/upload/xmyc/picture/2021-11/f5b04689-a2ee-4c2d-bdd4-f64a1d83ce38.png',
    '/upload/xmyc/picture/2021-11/f70ebf1b-c08f-46bf-bd1e-c45f7dfe1cd7.png',
    '/upload/xmyc/picture/202111/2021-11-19-10-18-34-11.png',
    '/upload/xmyc/picture/2021-11/2023f19b-0e44-4f93-bf0b-126263761dbf.png',
    '/upload/xmyc/picture/2021-11/1377ecd5-8bb2-4fae-8ec9-bf91de86c218.JPEG',
    '/upload/xmyc/picture/2021-11/35966546-dece-4de2-ac1d-04ef42ea081a.JPEG',
    '/upload/xmyc/picture/2021-11/e9217b15-082b-48d1-9257-744210dd2a52.JPEG',
    '/upload/xmyc/picture/2021-11/ca89b45d-4ce9-4b41-b83e-af8e7e61dfdd.png',
    '/upload/xmyc/picture/2021-12/14df9496-a35d-4334-9b3d-d10c80e97173.png',
    '/upload/xmyc/picture/2021-12/6285dba7-4574-4c8f-a469-6b3dc2af28a6.png',
    '/upload/xmyc/picture/202112/2021-12-01-12-18-28-11.png',
    '/upload/xmyc/picture/202112/2021-12-02-09-28-05-11.png',
    '/upload/xmyc/picture/2021-12/27b17e40-935e-4443-ae79-f66af6ebfcb1.png',
    '/upload/xmyc/picture/2021-12/e2933c1d-8824-4168-884d-96875bdad9d8.png',
    '/upload/xmyc/picture/2021-12/5d27d7bd-fac3-48f2-820a-4c0f03a0316e.png',
    '/upload/xmyc/picture/2021-12/a2f19b8a-6219-40f1-8215-f3d0efc2ee7d.png',
    '/upload/xmyc/picture/2021-12/97227091-29b9-4f2f-8dc2-24640e41b2fb.png',
    '/home/nqiv_admin/ycnqiv/samples/face/1.png',
    '/home/nqiv_admin/ycnqiv/samples/face/2.png',
    ]

final_paths = []
count = [0, 0, 0]

for i, fpath in enumerate(fpaths):
    if "crop" in fpath and '':
        continue
    data = {
        "pictureId":i,
        "screenshot": root_path + fpath,
        "isSource": 1,
    }

    data = json.dumps(data)

    r = requests.post(url, data)
    # print(r.text)
    feature = json.loads(r.text)
    is_save = feature["message"]["is_save"]
    feature = feature["message"]["feature"]

    count[is_save] += 1
    if is_save == 0:
        print(root_path + fpath)
        # img = cv2.imread(root_path + fpath)
        # cv2.imwrite('./test1202/' + fpath.split('/')[-1], img)
        final_paths.append(fpath)
        features.append(feature)

# print(count)
# print(len(features))
# exit()

# data = {
#     "pictureId":21,
#     "screenshot":'/upload/xmyc/picture/2021-11/76e72cfa-5f07-4811-a739-613164f3f7f0.png'
# }

# data = json.dumps(data)
# r = requests.post(url, data)
# print(r.text)

print(final_paths)

feature = features.pop(-1)
print(final_paths.pop(-1))

url = 'http://0.0.0.0:9090/feature_comparison'
data = {
        "feature": feature,
        "baseFeatures":[
            {
                "pictureId": i,
                "feature": features[i]
            } for i in range(len(features))
        ]
    }
data = json.dumps(data)

r = requests.post(url, data)
print(r.text)
r = json.loads(r.text)
for msg in r["message"]:
    if msg["matched"] != "0%":
        print(msg["matched"])
        print(final_paths[msg["pictureId"]])
