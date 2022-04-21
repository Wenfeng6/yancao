import timm
import torchvision.transforms as T
import numpy as np
import torch

from PIL import Image

from car_attribute.VehicleDC import Car_Classifier


def traffic_classifier(img, model):
    # model = timm.create_model('resnet50', pretrained=True)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        predict = model(img)

    class_num = [407, 408, 436, 444, 511, 555, 561, 569, 609, 654,
                 656, 665, 671, 675, 717, 734, 751, 779, 817, 829,
                 864, 867]
    class_name = ['救护车', '水陆两用车', '轿车', '双人自行车', '轿车', '消防车', '叉车', '卡车', '吉普车', '小客车',
                  '面包车', '摩托车', '自行车', '卡车', '轿车', '警车', '赛车', '校车', '轿车', '公交车',
                  '卡车', '自行车']
    result = []
    for ids in class_num:
        result.append(predict[0, ids].item())
    result = np.array(result)
    result = np.argmax(result)

    return result


def attribute_extractor(picname, net):
    classifier = net[0]
    img = Image.open(picname).convert('RGB')
    predict = traffic_classifier(img, net[1])
    # print(predict)
    carname = {'saloonCar':1, 'suv':1, 'van':1, 'passengerCar':2, 'truck':3}
    colorname = {'Red':'红色', 'Yellow':'黄色', 'Blue':'蓝色', 'Black':'黑色', \
        'Green':'绿色', 'Brown':'棕色', 'White':'白色', 'Gray':'灰色', 'Purple':'紫色'}
    if predict in [11]:
        return ('', '', 4)
    elif predict in [3, 12, 21]:
        return ('', '', 5)
    else:
        with torch.no_grad():
            ans = classifier.predict(img)
        
        ans = list(ans)
        if ans[2] in carname.keys():
            ans[2] = carname[ans[2]]
        if ans[0] in colorname.keys():
            ans[0] = colorname[ans[0]]

        return ans


if __name__ == '__main__':
    picname = '/home/nqiv_admin/ycnqiv/test.png'
    result = attribute_extractor(picname)
    print(result)
