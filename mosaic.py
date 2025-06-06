import cv2 as cv
import argparse
import time, sys
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

# parser 객체 생성
parser = argparse.ArgumentParser(description='Code for Cascade Classifier.')
# 사용할 인수 등록, 이름/타입/help (Optional Argument : 이름에 대시 두개 추가)
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
parser.add_argument('--video', help='Path to video', default='src/sample.mp4')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='data/SSD/face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='data/SSD/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
# 사용자에게 전달받은 인수를 args에 저장
args = parser.parse_args()
# args에 저장된 인수 사용 
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()


class Haar():
    def detectAndDisplay(frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        face_id = 1 # 얼굴 ID 시작 값
        for (x,y,w,h) in faces:
            faceROI_rec = [x-4, y-4, w+4, h+4]
            cv.rectangle(frame, faceROI_rec, (0, 255, 0), 4)
            # 얼굴 ID 출력 (face ID)
            cv.putText(frame, str(face_id), (faceROI_rec[0], faceROI_rec[1]), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            face_id += 1

            # 블러 처리 (kernel size = 20x20)
            faceROI = frame[y:y+h, x:x+w]
            faceROI = cv.blur(faceROI, (20,20), anchor=(-1,-1), borderType=cv.BORDER_DEFAULT)
            frame[y:y+h, x:x+w] = faceROI
        
        # cv.imshow('Capture - Face detection', frame)
        

    #-- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

class YuNet():
    def str2bool(v):
        if v.lower() in ['on', 'yes', 'true', 'y', 't']:
            return True
        elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
            return False
        else:
            raise NotImplementedError
    
    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                # 터미널창 감지된 얼굴 좌표 출력
                # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                # cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.putText(input, 'score: {:.2f}'.format(face[-1]), (coords[0]-thickness, coords[1]-thickness), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Frame per second
                
                # 블러 처리 (kernel size = 20x20)
                x, y, x_w, y_h = coords[0], coords[1], coords[0]+coords[2], coords[1]+coords[3]
                faceROI = input[y:y_h, x:x_w]
                faceROI = cv.blur(faceROI, (20,20), anchor=(-1,-1), borderType=cv.BORDER_DEFAULT)
                input[y:y_h, x:x_w] = faceROI
    
    def visualize_selectFace(input, faces, fps, exception=None, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                x2, y2 = x + w, y + h

                # 예외 좌표가 이 얼굴 영역 안에 있는지 확인
                if exception is not None and isinstance(exception, tuple) and len(exception) == 2:
                    ex_x, ex_y = exception
                    if x <= ex_x <= x2 and y <= ex_y <= y2:
                        print(f"Face {idx} 는 예외 처리 ")
                        continue  # idx에 해당되는 영역 블러 처리 건너뛰기

                # print(
                #     'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                #         idx, x, y, w, h, face[-1]))

                cv.rectangle(input, (x, y), (x2, y2), (0, 255, 0), thickness)
                cv.putText(input, 'score: {:.2f}'.format(face[-1]), (x - thickness, y - thickness),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 블러 처리
                faceROI = input[y:y2, x:x2]
                faceROI = cv.blur(faceROI, (20, 20), anchor=(-1, -1), borderType=cv.BORDER_DEFAULT)
                input[y:y2, x:x2] = faceROI

    def init():
        ## [initialize_FaceDetectorYN]
        detector = cv.FaceDetectorYN.create(
            args.face_detection_model,
            "",
            (320, 320),
            args.score_threshold,
            args.nms_threshold,
            args.top_k
        )
        ## [initialize_FaceDetectorYN]
        tm = cv.TickMeter()
        return tm, detector
        
    def display(cap, tm, detector, frame):
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
        detector.setInputSize([frameWidth, frameHeight])

        frame = cv.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()