from PIL import Image, ImageFilter, ImageOps
import numpy as np
import time
import tesserocr

def noise_remove_pil(img, k):
    def calculate_noise_count(img_obj, w, h):
        count = 0
        width, height = img_obj.size
        for _w_ in [w - 1, w, w + 1]:
            for _h_ in [h - 1, h, h + 1]:
                if _w_ > width - 1:
                    continue
                if _h_ > height - 1:
                    continue
                if _w_ == w and _h_ == h:
                    continue
                if img_obj.getpixel((_w_, _h_)) > 220:  # 这里因为是灰度图像，设置小于230为非白色 
                    count += 1
        return count

    gray_img = img.convert('L')

    w, h = gray_img.size
    for _w in range(w):
        for _h in range(h):
            if _w == 0 or _h == 0:
                gray_img.putpixel((_w, _h), 255)
                continue
            # 计算邻域非白色的个数
            pixel = gray_img.getpixel((_w, _h))
            if pixel == 255:
                continue
            if pixel > 80:
                gray_img.putpixel((_w, _h), 255)
            if calculate_noise_count(gray_img, _w, _h) > k:
                gray_img.putpixel((_w, _h), 255)
    gray_img.save('ocr_result1.png')
    return gray_img

def get_crop(picname):
    img = Image.open(picname).convert('RGB')
    # bbox = (1400, 20, 1860, 100)
    bbox = (1350, 10, 1900, 110)
    crop = img.crop(bbox)
    return crop

def preprocess(picname, thres=600):
    img = Image.open(picname).convert('RGB')
    # img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = img.filter(ImageFilter.SHARPEN())
    img = img.filter(ImageFilter.SHARPEN())
    # img = img.filter(ImageFilter.SHARPEN())
    img = np.array(img)

    H, W, _ = img.shape
    que = []
    visit = np.zeros_like(img[:, :, 0])

    for i in range(H):
        que.append((i, 0))
        visit[i, 0] = 1
        que.append((i, W-1))
        visit[i, W-1] = 1

    for i in range(W):
        que.append((0, i))
        visit[0, i] = 1
        que.append((H-1, i))
        visit[H-1, i] = 1

    margin = 255
    Dx = [0, 1, -1]
    Dy = [0, 1, -1]
    while len(que) > 0:
        (x, y) = que.pop(0)
        for dx in Dx:
            for dy in Dy:
                if dx == 0 and dy == 0:
                    continue
                tx = x + dx
                ty = y + dy
                if tx >= 0 and tx < H and ty >= 0 and ty < W:
                    if visit[tx, ty] == 0:
                        if np.abs(img[tx, ty] - img[x, y]).sum() < thres: # and np.abs(img[tx, ty] - img[x, y]).max() < 200:
                            que.append((tx, ty))
                            visit[tx, ty] = 1

    visit = visit.reshape(H, W, 1)
    img = img * (1 - visit)
    img = abs(255 - img)

    Image.fromarray(img).save('ocr_result.png')
    pilimg = Image.fromarray(img)
    return pilimg


def preprocess_test(picname, thres=500):
    import cv2
    img = Image.open(picname).convert('RGB')
    padimg = ImageOps.expand(img, border=(
        1, 1, 1, 1), fill=125)
    img = np.array(img)
    padimg = np.array(padimg)
    diff = np.zeros((9, *img.shape))
    print(diff.shape)
    Dx = [0, -1, 1]
    Dy = [0, -1, 1]
    H, W, _ = img.shape
    for ix, dx in enumerate(Dx):
        for iy, dy in enumerate(Dy):
            diff[ix*3 + iy] = padimg[dy+1:H+dy+1, dx+1:W+dx+1] - img
            # 0 0 -> 1:W+1, 1:H+1
            # 0 -1 -> 1:W+1, 0:H
    print(diff.shape)
    diffsum_channel = np.sum(diff, axis=-1)
    diffsum_direction = np.sum(diffsum_channel, axis=0)
    grad = np.array(np.where(diffsum_direction > thres*8, 0, 255))
    print(grad.shape)
    grad = grad.reshape(H, W, 1)
    # img = img * (1 - grad)
    # img = abs(255 - img)
    print(img.shape)
    img = grad
    # cv2.imwrite('ocr_result0.png', img)
    pilimg = Image.fromarray(img)
    return pilimg



def filter_text(text, flag):
    if flag == 0:
        filter_list = [str(i) for i in range(10)]
        filter_list.extend([':', '-', ' '])
        filter_text = ''.join((filter(lambda val: val in  filter_list, text)))
        while len(filter_text) > 0 and filter_text[0] in [' ', ':', '-']:
            filter_text = filter_text[1:]
        while len(filter_text) > 0 and filter_text[-1] in [' ', ':', '-']:
            filter_text = filter_text[:-1]
        yind = filter_text.find('-')
        if yind in [3, 5]:
            filter_text = '2021' + filter_text[yind:]
        elif yind == 4:
            if ' ' in filter_text[:yind]:
                filter_text = '2021' + filter_text[yind:]
    else:
        filter_list = [' ', '\n', '.', ',']
        filter_text = ''.join((filter(lambda val: val not in  filter_list, text)))
    return filter_text

def get_time(picname, flag=0):
    # get target location
    # crop = get_crop(picname)
    t0 = time.time()
    pilimg = preprocess(picname, thres = 550)
    t1 = time.time()
    print('preprocess:',t1 - t0)

    pilimg = noise_remove_pil(pilimg, 4)
    t2 = time.time()
    print('rel noise:',t2 - t1)
    # ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
    # pilimg = Image.fromarray(img)
    if flag:
        txt = tesserocr.image_to_text(pilimg, lang='chi_sim')
    else:
        txt = tesserocr.image_to_text(pilimg)
    t3 = time.time()
    print('ocr:',t3 - t2)
    txt = filter_text(txt, flag)
    t4 = time.time()
    print('filter:',t4 - t3)
    return txt


if __name__ == "__main__":
    pic0 = "samples/testocr_200.png"
    pic = "samples/testocrloc1.jpg"
    print(get_time(pic0, 0))
    # print(get_time(pic, 1))
