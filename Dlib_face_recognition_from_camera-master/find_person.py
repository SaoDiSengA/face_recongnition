# Author:Alvin
# Time:2021/11/22 15:38
import logging
import os

import cv2
import dlib
import pandas as pd

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
face_feature_known_list = []                # 用来存放所有录入人脸特征的数组 / Save the features of faces in database
face_name_known_list = []

def get_face_database(self):
    if os.path.exists("data/features_all.csv"):
        path_features_known_csv = "data/features_all.csv"
        csv_rd = pd.read_csv(path_features_known_csv, header=None)
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0, 128):
                if csv_rd.iloc[i][j] == '':
                    features_someone_arr.append('0')
                else:
                    features_someone_arr.append(csv_rd.iloc[i][j])
            face_feature_known_list.append(features_someone_arr)
            face_name_known_list.append("Person_" + str(i + 1))
        logging.info("Faces in Database：%d", len(self.face_feature_known_list))
        return 1
    else:
        logging.warning("'features_all.csv' not found!")
        logging.warning("Please run 'get_faces_from_camera.py' "
                        "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
        return 0


cap = cv2.VideoCapture(0)
print(cap.isOpened())
while cap.isOpened():
    ret_flag, img_camera = cap.read()
    faces = detector(img_camera, 0)  # Use Dlib face detector
    if len(faces) != 0:
        # 矩形框 / Show the ROI of faces
        for k, d in enumerate(faces):
            # 计算矩形框大小 / Compute the size of rectangle box
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            hh = int(height / 2)
            ww = int(width / 2)

            # 6. 判断人脸矩形框是否超出 480x640 / If the size of ROI > 480x640
            if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (d.left() - ww < 0) or (d.top() - hh < 0):
                cv2.putText(img_camera, "OUT OF RANGE", (20, 300), 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
            else:
                color_rectangle = (255, 255, 255)
                save_flag = 1

            cv2.rectangle(img_camera,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)
    cv2.imshow("camera", img_camera)
    # 每帧数据延时 1ms, 延时为0, 读取的是静态帧
    k = cv2.waitKey(1)
    # 按下 's' 保存截图
    if k == ord('s'):
        cv2.imwrite("test.jpg", img_camera)
    # 按下 'q' 退出
    if k == ord('q'):
        break
# 释放所有摄像头
cap.release()
# 删除建立的所有窗口
cv2.destroyAllWindows()
