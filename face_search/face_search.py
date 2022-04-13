# Author:Alvin
# Time:2021/11/29 21:45
import os
import cv2
import dlib
import numpy as np
import pandas as pd

detector = dlib.get_frontal_face_detector()  # 正脸监测器 返回值是一个矩形

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

test_face_feature_list = []
db_face_feature_list = []

def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


img_face = cv2.imread("test-face/test.jpeg")
test_faces = detector(img_face, 0)  # 第二个参数   代表将原始图像是否进行放大，1表示放大1倍再检查，提高小人脸的检测效果 <class 'dlib.dlib.rectangles'>
print(type(test_faces))
if len(test_faces) != 0:
    for i in range(len(test_faces)):
        shape = predictor(img_face, test_faces[i]) #img：一个numpy ndarray，包含8位灰度或RGB图像   box：开始内部形状预测的边界框   返回值：68个关键点的位置
        print(type(predictor)) # <class 'dlib.dlib.shape_predictor'>
        test_face_feature_list.append(face_reco_model.compute_face_descriptor(img_face, shape))
        #功能：图像中的68个关键点转换为128D面部描述符，其中同一人的图片被映射到彼此附近，并且不同人的图片被远离地映射。
        #img_rd：人脸灰度图，类型：numpy.ndarray
        #shape：68个关键点位置 返回值：128D面部描述符
        print(type(face_reco_model.compute_face_descriptor(img_face, shape)))
        person_list = os.listdir("sources/")
        # person_num_list = []
        # for person in person_list:
        #     person_num_list.append(int(person.split('_')[-1]))
        # person_cnt = max(person_num_list)
        # for person in range(person_cnt):
        for filename in os.listdir("sources"):
            print(filename)
            img = cv2.imread("sources" + "/" + filename)
            db_faces = detector(img, 0)
            if len(test_faces) != 0:
                for i in range(len(db_faces)):
                    shape = predictor(img, db_faces[i])  # img：一个numpy ndarray，包含8位灰度或RGB图像   box：开始内部形状预测的边界框   返回值：68个关键点的位置
                    print(type(predictor))  # <class 'dlib.dlib.shape_predictor'>
                    db_face_feature_list.append(face_reco_model.compute_face_descriptor(img, shape))
            for i in range(len(db_face_feature_list)):
                e_distance = return_euclidean_distance(test_face_feature_list[i],db_face_feature_list[i])
                if e_distance < 0.4:
                    cv2.imshow("123", img)
                    cv2.waitKey(0)
cv2.imshow("dectect", img_camera)
k = cv2.waitKey(1)  # 每帧图像显示演示1ms
# 释放所有摄像头
cap.release()
# 删除建立的所有窗口
cv2.destroyAllWindows()
