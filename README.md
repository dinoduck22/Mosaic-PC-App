![header](https://capsule-render.vercel.app/api?type=transparent&text=Video_Mosaic(Blur)&fontColor=FFD66B)

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyQt](https://img.shields.io/badge/-PyQt-blue) ![OpenCV](https://img.shields.io/badge/-OpenCV-Green)

# Overview
얼굴 자동 검출 모자이크 처리

# Summary
- 영상 파일(mp4) / 실시간 카메라
- HaarCascade / YuNet(DNN) 기법
- 모자이크 처리 확인 / 저장
- 단일 얼굴 선택 모자이크 해제 / 설정

# Preview
<img src="src/preview.png" alt="preview" width="460" height = "400"/>

# How To Use
``` 
git clone https://github.com/dinoduck22/Mosaic-PC-App.git
cd Mosaic-PC-App
python main.py 
```

# Reference
- OpenCV: Cascade Classifier, (https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- OpenCV: Smoothing Images, (https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- How to display opencv video in pyqt apps, Gist, 
(https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1)
- Face_detect.py, OpenCV, Git-Hub, (https://github.com/opencv/opencv/blob/master/samples/dnn/face_detect.py)
- PyQt5 쓰레드로 동영상 재생 제어하기, https://toyourlight.tistory.com/122?category=1468376
