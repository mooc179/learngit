from ctypes import *
import math
import random
import os
import cv2
import time


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


# wjh  将ndarray数组转换为Image对象
def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("/home/xyh/darknet/python_xiangmu/niaocao/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/madridista/PycharmProjects/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

# wjh
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def init1():
    net = load_net("/home/madridista/PycharmProjects/darknet/xiangmu/nest/FirstStep/20191229/yolov3-voc.cfg".encode('utf-8'),
                   "/home/madridista/PycharmProjects/darknet/xiangmu/nest/FirstStep/20191229/yolov3-voc_20000.weights".encode('utf-8'),
                   0)
    meta = load_meta("/home/madridista/PycharmProjects/darknet/xiangmu/nest/FirstStep/20191229/voc.data".encode('utf-8'))
    return net, meta


def init2(cfg, weights, vocdata):
    net = load_net(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
    meta = load_meta(vocdata.encode('utf-8'))
    return net, meta


def cut(r, img):
    kinds = []
    regions = []
    coordinates = []
    h, w = img.shape[:2]
    for item in r:
        objName = item[0].decode()
        if objName == 'nest1' or objName == "nest2" or objName == 'nest3' or objName == 'nest4' or objName == 'nest5' \
                or objName == "nest6":
        # if objName == 'nest4':
            scores = item[1]
            box = item[2]
            show_x = int(box[0])
            show_y = int(box[1])
            show_w = int(box[2])
            show_h = int(box[3])
            x_min = 2 if ((show_x - show_w // 2) < 2) else (show_x - show_w // 2)
            y_min = 2 if ((show_y - show_h // 2) < 2) else (show_y - show_h // 2)
            x_max = 2558 if ((show_x + show_w // 2) > 2558) else (show_x + show_w // 2)
            y_max = 2158 if ((show_y + show_h // 2) > 2158) else (show_y + show_h // 2)
            img_cut = img[y_min: y_max, x_min: x_max]
            coordinate = (x_min, y_min, x_max, y_max)
            kinds.append(objName)
            regions.append(img_cut)
            coordinates.append(coordinate)
    return kinds, regions, coordinates

def draw_img(img, has_nests, coordinate, img_path, rlt_path):
    path = img_path.split('/')[-1]
    h, w = img.shape[:2]
    thick = int((h + w) / 500)  # 设置框的粗细
    for i, has_nest in enumerate(has_nests, 0):
        if has_nest:
            x_min = coordinate[i][0]
            y_min = coordinate[i][1]
            x_max = coordinate[i][2]
            y_max = coordinate[i][3]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thick // 3)
            # cv2.putText(img, 'nest', (x_min, y_min), 1, 1e-3 * w, (0, 0, 255), thick // 3)
    cv2.imwrite(os.path.join(rlt_path, path), img)

nest_model = {"nest1_2_3_5_6": ["/home/madridista/PycharmProjects/darknet/xiangmu/nest/SecondStep/nest1_2_3_5_6/yolov3-voc.cfg",
                              "/home/madridista/PycharmProjects/darknet/xiangmu/nest/SecondStep/nest1_2_3_5_6/20191227x/yolov3-voc.backup",
                              "/home/madridista/PycharmProjects/darknet/xiangmu/nest/SecondStep/nest1_2_3_5_6/niaocao.data"]
              }

class Detect(object):
    def __init__(self):
        self.net1, self.meta1 = init1()
        self.net2_path = nest_model["nest1_2_3_5_6"]
        self.net2, self.meta2 = init2(self.net2_path[0], self.net2_path[1], self.net2_path[2])

    def run(self, test_path, rlt_path):
        if not os.path.exists(rlt_path):
            os.mkdir(rlt_path)
        start = time.time()
        img_has_nests = 0
        for path in sorted(os.listdir(test_path)):
            img_path = os.path.join(test_path, path)
            img = cv2.imread(img_path)
            print(img_path)

            r1 = detect(self.net1, self.meta1, nparray_to_image(img), thresh=.5)
            print(r1)
            kinds, regions, coordinates = cut(r1, img) # nest种类，nest区域图像，nest框的坐标
            has_nests = []

            if len(kinds) != 0:
                for i in range(len(kinds)):                     # 对每个部位进行鸟窝检测
                    has_nest = False
                    if kinds[i] == 'nest1':
                        # r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.65)
                        r2 = [1, ] # 将所有隔离开关均识别为鸟窝
                    elif kinds[i] == 'nest2':
                        r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.65)
                    elif kinds[i] == 'nest3':
                        r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.5)
                    elif kinds[i] == 'nest4':
                        r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.8)
                    elif kinds[i] == 'nest5':
                        r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.8)
                    elif kinds[i] == 'nest6':
                        r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.8)
                    elif kinds[i] == 'nest7':
                        r2 = detect(self.net2, self.meta2, nparray_to_image(regions[i]), thresh=.5)
                    # print(r2)
                    if len(r2) != 0:
                        has_nest = True
                    has_nests.append(has_nest)

            if True in has_nests:             # 如果has_nests中有True，说明该图中有鸟窝
                img_has_nests += 1
            print(has_nests)
            draw_img(img, has_nests, coordinates, img_path, rlt_path)

        print(img_has_nests)
        end = time.time()
        print('total time is {:.2f}s, every/home/madridista/Desktop/nest2/test/0 image time is {:.2f}s.'.format(end-start, (end-start)/len(os.listdir(test_path))))


if __name__ == "__main__":
    test_path = '/home/madridista/Desktop/自测图片/20190314京广高铁'
    rlt_path = '/home/madridista/Desktop/自测图片/20190314京广高铁_out'
    detector = Detect()
    detector.run(test_path, rlt_path)