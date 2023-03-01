# -*- coding:utf-8 -*-
# @Time : 2021/9/25
# @Author: YY0628
# @File : label_utils.py
# @Email : yinyong@mail.ustc.edu.cn

import os
import random
from shutil import copy
from xml.etree.ElementTree import parse


def get_labels(path) -> list:
    """
    通过标签txt文件，返回label列表\n
    :param path: txt文件目录，可从 H:\\Program Files\\labelImg\\data\\predefined_classes.txt 复制
    :return: label列表
    """
    labels = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            labels.append(line.strip())
    return labels


class XmlTransfer:
    """
        Xml文件格式变换
    """

    def __init__(self, xml_dir, labelName_path, selected_classes=None):
        """
        :param xml_dir: 包含xml文件夹的目录
        :param labelName_path: label标签txt文件目录
        :param selected_classes: 选中训练的种类，用 predefined_classes.txt 中的种类，自上而下，从0编号
        """
        self.xml_path = xml_dir
        self.labelName_path = labelName_path
        self.selected_classes = selected_classes
        self.selected_classes_nums = None

    def choose_class(self, selected_classes=None, saved_path=None):
        """
        选择需要训练的种类\n
        :param selected_classes: 选择要训练的种类
        :param saved_path: 选择新的label文件保存的路径
        :return:
        """
        self.selected_classes = selected_classes
        if selected_classes == None:
            self.selected_classes = [0 for i in range(len(get_labels(self.labelName_path)))]

        self.selected_classes_nums = [0 for i in range(len(self.selected_classes))]
        labels = get_labels(self.labelName_path)

        if saved_path is None:
            saved_path = "chosen_class.txt"
        with open(saved_path, 'w') as file:
            for i in self.selected_classes:
                file.write(str(labels[i]) + "\n")

    @staticmethod
    def convert_darknet(size, box):
        """
        将标签中的值 (xmin, xmax, ymin, ymax) 转为darknet所需 (cx,cy,w,h) \n
        :param size: 图片大小 (width, height)
        :param box: xml标注(xmin, xmax, ymin, ymax)
        :return: (cx, cy, width, height)
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = abs(box[1] - box[0])
        h = abs(box[3] - box[2])
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h


    # 如果xml标签内的body被标记为difficult 则不选取它作为数据集
    def xml_to_txt_darknet_withoutBodyDiff(self, txt_dir, bodyDiff=True):
        """
        Darknet制作数据集方式\n
        将xml文件转为txt文件
        可以通过设置label文件选取参与训练的种类
        不同的训练模型会对标签文件有不同的要求，Darknet为：\n

        :param txt_dir: label标签txt文件的文件夹
        :return: txt文件数\n

                                                               类型   左上角x       左上角y      右下角x       右下角y
        frosted_c21.xml   -------->    frosted_c21.txt -------- 2 0.0164473684 0.2549342105 0.9851973684 0.7434210526
                                                       -------- 3 0.9621710526 0.7319078947 0.9851973684 0.7483552632
        frosted_c41.xml   -------->    frosted_c41.txt -------- 2 0.0148026316 0.2549342105 0.9835526316 0.7434210526
                                                       -------- 3 0.9588815789 0.25         0.9851973684 0.28125
                                                       -------- 5 0.6019736842 0.2516447368 0.6282894737 0.2680921053
                                                       -------- 7 0.2993421053 0.2483552632 0.3174342105 0.2664473684
                                                       -------- 9 0.8092105263 0.375        0.8355263158 0.3898026316
        frosted_c45.xml   -------->    frosted_c45.txt -------- 2 0.0625       0.2779605263 0.9440789474 0.7220394737
                                                       -------- 3 0.9194078947 0.6973684211 0.9457236842 0.7253289474
                                                       -------- 3 0.9194078947 0.6973684211 0.9457236842 0.7253289474
                                                       -------- 3 0.9292763158 0.2763157895 0.9424342105 0.2944078947
        frosted_c60.xml   -------->    frosted_c60.txt -------- 2 0.0476973684 0.2730263158 0.9523026316 0.7286184211
                                                       -------- 5 0.3289473684 0.2697368421 0.3766447368 0.2845394737

             .......                        ......\n

        """
        if not os.path.exists(txt_dir):
            os.mkdir(txt_dir)
        

        # 获取xml文件的绝对路径列表
        xml_list = os.listdir(self.xml_path)
        cnt = len(xml_list)
        for i in range(len(xml_list)):
            xml_list[i] = os.path.join(self.xml_path, xml_list[i])

        # 获取标签种类列表
        labels = get_labels(self.labelName_path)
        if self.selected_classes is not None:
            labels = [labels[x] for x in self.selected_classes]

        # 遍历xml文件路劲列表进行操作
        for file in xml_list:
            tree = parse(file)
            root = tree.getroot()
            L = ''

            # 如果xml标签内的body被标记为difficult 则不选取它作为数据集
            if XmlTransfer.isBodyDiff(root):
                continue

            # 获取并重命名文件名称
            filename = root.find('filename').text
            filename = str(filename)[:-4] + ".txt"

            # 获取图像大小信息
            size = root.find("size")
            width = eval(size.find("width").text)
            height = eval(size.find("height").text)

            # 获取bonding box信息
            for obj in root.iter('object'):
                name = obj.find('name').text

                # 选取label文件中的种类标签作为训练种类
                if name in labels:
                    class_num = labels.index(name)
                    self.selected_classes_nums[labels.index(name)] += 1
                else:
                    continue

                xmlbox = obj.find('bndbox')
                xmin = eval(xmlbox.find('xmin').text)
                xmax = eval(xmlbox.find('xmax').text)
                ymin = eval(xmlbox.find('ymin').text)
                ymax = eval(xmlbox.find('ymax').text)

                cx, cy, w, h = XmlTransfer.convert_darknet((width, height), (xmin, xmax, ymin, ymax))
                L += str(class_num) + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + "\n"

            txt_path = os.path.join(txt_dir, filename)
            with open(txt_path, 'w') as f:
                f.write(L)
                print(txt_path + " succeed")
        return cnt

    # 判断标签内的body是否被标记为difficult
    @staticmethod
    def isBodyDiff(root):
        for obj in root.iter("object"):
            name = obj.find("name").text
            diff = obj.find("difficult").text
            if name == "body" and diff == "1":
                return True
        return False
    
    # 判断标签内是否被标记为difficult
    @staticmethod
    def isAnyDiff(root):
        for obj in root.iter("object"):
            diff = obj.find("difficult").text
            if diff == "1":
                return True
        return False
    
    # 如果xml标签内只要有被标记为difficult的obj就不选取
    def xml_to_txt_darknet_withoutAnyiff(self, txt_dir, bodyDiff=True):
        pass




class DataPartition:
    """
        数据集划分
    """

    def __init__(self, txt_path, train=0.8, validation=0.2, test=0.0):
        self.txt_path = txt_path
        self.train = train
        self.validation = validation
        self.test = test

    def set_proportion(self, train, validation, test=0.0):
        """
        划分训练集、测试集、验证集的比例\n
        :param train: 训练集比例
        :param validation: 验证集比例
        :param test: 测试集比例
        :return: list [训练集数据个数, 验证集数据个数, 测试集数据个数]
        """
        self.train = train
        self.validation = validation
        self.test = test

    def split_data_darknet(self, data_dir, train_path, validation_path, test_path=None):
        """
        划分数据集，一个txt文件含有所有同种数据集类型图片信息\n
        :param data_dir: 数据集文件所存放的文件夹
        :param train_path: 训练集文件
        :param validation_path: 验证集文件
        :param test_path: 测试集文件
        :return: 训练集、验证集、测试集个数
        """
        annotations = os.listdir(self.txt_path)
        random.seed()
        train_list = []
        validation_list = []
        test_list = []

        for file in annotations:
            x = random.random()
            if x <= self.train:
                train_list.append(data_dir + '/' + file[:-4] + ".bmp" + '\n')
            elif self.train < x <= self.train + self.validation:
                validation_list.append(data_dir + '/' + file[:-4] + ".bmp" + '\n')
            else:
                test_list.append(data_dir + '/' + file[:-4] + ".bmp" + '\n')

        with open(train_path, 'w') as train_file, open(validation_path, 'w') as validation_file:
            train_file.writelines(train_list)
            validation_file.writelines(validation_list)
        if test_path is not None:
            with open(test_path, 'w') as test_file:
                test_file.writelines(test_list)

        return len(train_list), len(validation_list), len(test_list)

    def split_data_v2(self, train_dir, validation_dir, test_dir=None):
        """
        划分数据集，一个txt文件对应一张图片，一个文件夹对应数据集划分\n
        :param train_dir: 训练集存放文件夹
        :param validation_dir: 验证集存放文件夹
        :param test_dir: 测试集存放文件夹
        :return: 训练集、验证集、测试集个数

        train------frosted_c21.txt     validation------frosted_r9.txt
             ------frosted_c41.txt               ------frosted_r15.txt
             ------frosted_c45.txt               ------frosted_r40.txt
             ------frosted_c60.txt               ------frosted_r113.txt
             ......                              ......
        """
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)
        if test_dir is not None and not os.path.exists(test_dir):
            os.mkdir(test_dir)

        annotations = os.listdir(self.txt_path)
        random.seed()
        cnt_train = 0
        cnt_validation = 0
        cnt_test = 0
        for file in annotations:
            x = random.random()
            from_path = os.path.join(str(self.txt_path), file)
            if x <= self.train:
                to_path = os.path.join(str(train_dir), file)
                cnt_train += 1
            elif self.train < x <= self.train + self.validation:
                to_path = os.path.join(str(validation_dir), file)
                cnt_validation += 1
            else:
                to_path = os.path.join(str(test_dir), file)
                cnt_test += 1
            copy(from_path, to_path)

        return cnt_train, cnt_validation, cnt_test


if __name__ == "__main__":

    # xml 转为 txt 并选取合法的数据加入数据集
    # data = XmlTransfer("dataset/labels/labels_big_dot_xml", "dataset/names.txt")
    # data.choose_class([0,1])
    # data.xml_to_txt_darknet_withoutBodyDiff("dataset/labels/labels_big_dot_yolo")

    # 划分数据集
    data = DataPartition("dataset/labels/labels_big_dot_yolo")
    data.split_data_darknet("dataset/images", "dataset/train.txt", "dataset/validation.txt")


