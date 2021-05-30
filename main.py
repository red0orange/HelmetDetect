import os
import cv2
import numpy as np


if __name__ == '__main__':
    use_image = False
    image_path = "eyes_template.png"

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        if use_image:
            frame = cv2.imread(image_path)
        else:
            ret, frame = cap.read()
            if not ret:
                print("error read video")
                continue
        ori_image = frame
        draw_image = ori_image.copy()
        result_draw_image = ori_image.copy()

        faces = face_cascade.detectMultiScale(
            ori_image,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40)
        )
        eyes = eye_cascade.detectMultiScale(ori_image, 1.3, 10)  # return list, per item format is [x, y, w, h]
        # print(eyes)

        # draw face
        for (x, y, w, h) in faces:
            # 画出人脸框，蓝色，画笔宽度微
            img = cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(draw_image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        layers = cv2.split(ori_image)
        feature_layer = cv2.subtract(layers[-1], layers[0])
        _, feature_layer = cv2.threshold(feature_layer, 80, 255, cv2.THRESH_BINARY)

        # 开始检测是否有有效的人脸或眼睛对, 用人工特征
        valid_faces = []
        valid_eye_pairs = []
        if len(eyes) < 2 and len(faces) == 0:
            pass
        else:
            if len(eyes) >= 2:
                # 用人工特征检测是否有两个有效的眼睛，如果有就说明检测成功
                sort_eyes = sorted(eyes, key=lambda i: i[0] + i[2] // 2)
                eyes_center = [(i[0] + i[2] // 2, i[1] + i[3] // 2) for i in sort_eyes]
                for i in range(len(eyes_center) - 1):
                    l_eye = sort_eyes[i]
                    l_eye_center = eyes_center[i]
                    for j in range(i+1, len(eyes_center)):
                        r_eye = sort_eyes[j]
                        r_eye_center = eyes_center[j]

                        area_diff = abs(r_eye[2] * r_eye[3] - l_eye[2] * l_eye[3]) / max(r_eye[2] * r_eye[3], l_eye[2] * l_eye[3])
                        y_diff = abs(r_eye_center[1] - l_eye_center[1])
                        x_diff = r_eye_center[0] - l_eye_center[0]

                        # print(area_diff, x_diff, y_diff)
                        if area_diff < 0.6 and x_diff > 60 and y_diff < 20:
                            valid_eye_pairs.append([l_eye, r_eye])
                        pass
                # print(len(valid_eye_pairs))
                pass

            if not len(valid_eye_pairs) and len(faces):
                # TODO 人脸检测的信息也要用
                pass

        # 开始检测头盔
        if not len(valid_eye_pairs) and not len(valid_faces):
            print("error detect person face!")
        else:
            # 开始检测红色的头盔是否在头顶
            if len(valid_eye_pairs):
                # 优先使用eye pair
                l_eye, r_eye = valid_eye_pairs[0]  # TODO 改为选取最匹配的
                l_eye_center = (l_eye[0] + l_eye[2] // 2, l_eye[1] + l_eye[3] // 2)
                r_eye_center = (r_eye[0] + r_eye[2] // 2, r_eye[1] + r_eye[3] // 2)
                eyebrow = ((l_eye_center[0] + r_eye_center[0]) // 2, (l_eye_center[1] + r_eye_center[1]) // 2)
                direct_distance = ((l_eye_center[0] - r_eye_center[0]) ** 2 + (l_eye_center[1] - r_eye_center[1]) ** 2) ** (1 / 2)
                helmet_center = (eyebrow[0], int(eyebrow[1] - direct_distance * 1.05))
                cv2.ellipse(result_draw_image, helmet_center, (150, 120), 0, 180, 360, color=(255, 255, 0), thickness=2)
                # 获得特征层红色的占头上一块区域的比例
                mask = np.zeros(ori_image.shape[:2], dtype=np.uint8)
                cv2.ellipse(mask, helmet_center, (150, 120), 0, 180, 360, color=255, thickness=-1)
                cv2.imshow("test", mask)
                roi_pixel_num = np.sum(mask == 255)
                roi_valid_binary = cv2.bitwise_and(feature_layer, mask)
                cv2.imshow("test1", roi_valid_binary)
                roi_valid_pixel_num = np.sum(roi_valid_binary == 255)
                valid_ratio = roi_valid_pixel_num / roi_pixel_num
                print(valid_ratio)
                if valid_ratio > 0.3:
                    print("have helmet !")
                else:
                    print("no have helmet !")
                pass
            elif len(valid_faces):
                # TODO 人脸检测的信息也要用
                pass

        # draw result valid face or eyes
        # draw face
        for (x, y, w, h) in valid_faces:
            # 画出人脸框，蓝色，画笔宽度微
            cv2.rectangle(result_draw_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # draw eyes
        for eye_pair in valid_eye_pairs:
            for (ex, ey, ew, eh) in eye_pair:
                cv2.rectangle(result_draw_image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        cv2.imshow("binary", feature_layer)
        cv2.imshow("detect", draw_image)
        cv2.imshow("result", result_draw_image)
        key = cv2.waitKey(3)

        if len(eyes) >= 2:
            cv2.imwrite("test_success.png", ori_image)
        if key == ord("p"):
            cv2.imwrite("test.png", ori_image)
