from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from matplotlib import pyplot as plt
import time, sys
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QPoint
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QRadioButton, QStatusBar, QAction, QFileDialog, QMessageBox, QProgressDialog
from mosaic import args, Haar, YuNet
from queue import Queue
from threading import Thread
import os

#QLabel 상속 받아서 마우스 이벤트 연결을 위한 추상 클래스
class ClickableLabel(QLabel):
    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, event):
        self.clicked.emit(event.pos())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()        
        # 타이틀
        self.setWindowTitle("실시간 얼굴 감지 및 모자이크 처리")
        # 위젯
        widget = App()
        # 메뉴바 생성
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File') # '&' 단축키 지정 (Alt+F)
        # 액션 설정
        openFile = QAction('Select File', self)
        openFile.setStatusTip('Create a new file') # 상태바에 표시될 팁
        openFile.triggered.connect(lambda: self.select_file(widget)) # 연결 설정
        file_menu.addAction(openFile)
        # 상태바 생성
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.setCentralWidget(widget)

    def select_file(self, widget):
        fname = QFileDialog.getOpenFileName(self, "Choose Video", './src/', '*.mp4')
        # 사용자가 파일 선택을 취소한 경우
        if not fname[0]:  # fname이 빈 문자열일 경우
            self.status_bar.showMessage("파일 선택이 취소되었습니다.")
            return  # 아무 작업도 하지 않고 함수 종료
        else :
            args.video = fname[0]
            self.status_bar.showMessage("파일선택 : %s" % fname[0])
            widget.set_cap(args.video)
            widget.queue_video()
            

class App(QWidget):
    def set_cap(self, name):
        self.cap = cv.VideoCapture(name)
        if not self.cap.isOpened():  # 동영상이 제대로 열리지 않은 경우
            window.status_bar.showMessage("동영상을 불러오는 데 실패하였습니다.")
            return  # 에러 처리 후 함수 종료

    def queue_video(self):
        width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.image_label.resize(int(width), int(height))
        ret, img = self.cap.read() # 프레임 읽기
        if ret:
            self.queue.put(img) # 첫 번째 프레임을 큐에 넣음
            self.disp_video() #비디오 첫 화면만 보여주기
        else:
            print("failed")

    def disp_video(self):
        if not self.queue.empty():
            image = self.queue.get()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            h, w, c = image.shape
            qImg = QImage(image.data, w, h, w*c, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qImg)
            self.image_label.setPixmap(self.pixmap)    
            QApplication.processEvents()

    def __init__(self):
        super().__init__()
        # 변수
        self.worker_thread = None # 쓰레드 객체 정의
        self.queue = Queue() # 동영상 프레임 큐
        self.is_playing = False # 재생 확인
        self.is_paused = False # 일시정지 상태 확인
        self.live = False # 라이브 버튼 클릭
        self.cap = None # 동영상 재생
        self.mode = None # 모자이크 모드
        self.writer = None # 비디오 저장
        self.click=[] #마우스 클릭 관련 변수
        # Width, Height
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('컴퓨터비전기반오토모티브SW(01)-2025.06.06-유하영(22114118)')
        # 버튼 생성
        self.btnLive = QPushButton('Live Camera')
        self.btnReset = QPushButton('Reset')
        self.btnStop = QPushButton('⏸️ Stop')
        self.btnPlay = QPushButton('▶️ Play')
        self.btnSave = QPushButton('✅ Save')
        self.btnY = QRadioButton('YuNet(DNN)')
        self.btnH = QRadioButton('HaarCascade')
        # 버튼 이벤트
        self.btnLive.clicked.connect(self.on_btnLive_clicked)
        self.btnReset.clicked.connect(self.on_btnReset_clicked)
        self.btnStop.clicked.connect(self.on_btnStop_clicked)
        self.btnPlay.clicked.connect(self.on_btnPlay_clicked)
        self.btnSave.clicked.connect(self.on_btnSave_clicked)
        self.btnH.clicked.connect(self.on_btnH_clicked)
        self.btnY.clicked.connect(self.on_btnY_clicked)
        # 이미지 이벤트
        self.image_label = ClickableLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        #마우스 클릭 연결
        self.image_label.clicked.connect(self.mouse_click)
        
        # 가로 레이아웃 생성1
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.btnStop)
        hbox1.addWidget(self.btnPlay)
        hbox1.addWidget(self.btnSave)

        # 가로 레이아웃 생성2
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.btnH)
        hbox2.addWidget(self.btnY)
        hbox2.addWidget(self.btnReset)
        
        # 세로 레이아웃 생성 and add the two labels
        vbox = QVBoxLayout()
        # vbox.addWidget(self.btnStart)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.btnLive)
        vbox.addWidget(self.image_label)
        vbox.addLayout(hbox1)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
    
    # 초기화
    def on_btnReset_clicked(self):
        self.mode = None
        self.btnH.setChecked(False)
        self.is_playing = False
        self.is_paused = False
        self.live = False
        self.image_label.clear()
        window.status_bar.showMessage("Reset")
        if self.worker_thread is not None:
            self.worker_thread.join()

    # 라이브 버튼 클릭 이벤트
    def on_btnLive_clicked(self):
        if not self.is_playing:  # 재생 중이 아닐 때만 실행
            # 기존 영상 재생 중이면 중지
            if self.cap:
                self.cap.release()
            
            self.cap = cv.VideoCapture(0)  # 0번 카메라 (웹캠)
            if not self.cap.isOpened():
                window.status_bar.showMessage("카메라를 열 수 없습니다.")
                return

            self.mode = None  # mode 초기화
            self.queue_video()
            self.thread_worker()
            self.live = True
            if self.live:
                self.btnLive.setDisabled(True)
            window.status_bar.showMessage("실시간 카메라 모드 시작")
    
    # 정지 버튼 클릭 이벤트
    def on_btnStop_clicked(self):
        if self.live == True:
            self.live = False
            self.btnLive.setDisabled(False)
        if self.is_playing:
            self.is_paused = not self.is_paused  # 일시정지 상태 토글
            window.status_bar.showMessage("Stopped")
                

    def thread_worker(self):
        while self.is_playing:
            if not self.is_paused: # 일시정지 상태가 아닐 때만 프레임을 읽음
                fps = self.cap.get(cv.CAP_PROP_FPS)
                ret, img = self.cap.read() # 프레임 읽기
                delay = 0.5 / fps if fps > 0 else 0.03
                if self.mode == 'HaarCascade mode' :
                    Haar.detectAndDisplay(img) # 모델 추가
                    delay = 0.0003
                elif self.mode == 'YuNet mode' :
                    tm, detector = YuNet.init()
                    YuNet.display(self.cap, tm, detector, img)
                    faces = detector.detect(img)

                    #얼굴 선택
                    if self.click is not None: #클릭 했을 시
                        exception_point = self.click  # 마지막 클릭한 좌표
                        # print(f"클릭함 {exception_point}")
                    else:
                        exception_point = None

                    # Draw results on the input image, 제외 범위 추가 전달
                    delay = 0.0003
                    # YuNet.visualize(img, faces, tm.getFPS())
                    YuNet.visualize_selectFace(img, faces, tm.getFPS(), exception_point)

                if ret:
                    self.queue.put(img)
                    self.disp_video() #비디오 재생하기
                    time.sleep(delay)
                else:
                    window.status_bar.showMessage("Finished")
                    break
            else:
                time.sleep(0.1)  # 일시정지 상태일 때 대기
        self.cap.release()

    # 재생 버튼 클릭 이벤트
    def on_btnPlay_clicked(self):
        self.is_paused = False  # 재생 상태로 설정
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.is_playing = True
            self.worker_thread = Thread(target=self.thread_worker, daemon=True)
            self.worker_thread.start()
    
    # 저장 버튼 클릭 이벤트
    def on_btnSave_clicked(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "경고", "저장할 영상이 없습니다.")
            return

        reply = QMessageBox.question(
            self, '영상 저장', '모자이크 처리된 영상을 저장하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.save_video()

    def save_video(self):
        total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0  # 예외 처리
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print(width, height)

        # 비디오 저장 경로
        save_path, _ = QFileDialog.getSaveFileName(self, "영상 저장", "./mosaic_output.mp4", "MP4 Files (*.mp4)")
        if not save_path:
            return

        # 프로그레스 다이얼로그 생성
        progress = QProgressDialog("영상 저장 중...", "취소", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # 비디오 라이터 객체 생성
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(save_path, fourcc, fps, (width, height))

        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # 영상 처음부터 시작

        frame_idx = 0
        cancelled = False
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 얼굴 감지 및 모자이크 처리
            if self.mode == 'HaarCascade mode':
                Haar.detectAndDisplay(frame)
            elif self.mode == 'YuNet mode':
                tm, detector = YuNet.init()
                YuNet.display(self.cap, tm, detector, frame)
                faces = detector.detect(frame)
                YuNet.visualize(frame, faces, tm.getFPS())

            out.write(frame)

            frame_idx += 1
            progress.setValue(frame_idx)
            QApplication.processEvents()

            if progress.wasCanceled():
                cancelled = True
                break

        out.release()
        progress.close()

        if cancelled:
            # # 저장 취소 시 파일 삭제
            # if os.path.exists(save_path):
            #     os.remove(save_path)
            QMessageBox.information(self, "저장 취소", "영상 저장이 취소되었습니다.")
        else:
            QMessageBox.information(self, "저장 완료", "영상 저장이 완료되었습니다.")

        
    # Haar 버튼 클릭 이벤트
    def on_btnH_clicked(self):
        self.mode = 'HaarCascade mode'
        self.textLabel.setText(self.mode)

    # YuNet 버튼 클릭 이벤트
    def on_btnY_clicked(self):
        self.mode = 'YuNet mode'
        self.textLabel.setText(self.mode)
    
    #영상 나오는 화면 클릭 이벤트
    def mouse_click(self,pos):
        self.click= (pos.x(),pos.y())
        print(f"클릭한 위치: {pos.x()}, {pos.y()}")
        print(f"저장된 {self.click}")
        print(f"클릭 X: {self.click[0]}")

# 메인
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())