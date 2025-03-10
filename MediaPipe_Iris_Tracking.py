import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pydub import AudioSegment
from pydub.playback import play

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 設定視窗標題和大小
        self.setWindowTitle("人機介面")
        self.setGeometry(100, 100, 800, 600)
        
        # 建立攝影機畫面和狀態顯示的 QLabel 元件
        self.video_label = QLabel(self)
        self.status_label = QLabel(self)
        
        # 設定狀態顯示的字型和大小
        font = QFont()
        font.setPointSize(15)
        self.status_label.setFont(font)
        self.status_label.setAlignment(Qt.AlignCenter)

        
        # 設定攝影機畫面的位置和大小
        self.video_label.setGeometry(0, 0, 640, 480)
        
        # 設定狀態顯示的位置和大小
        self.status_label.setGeometry(0, 480, 800, 120)
        
        # 設定攝影機和提示音檔案的路徑
        self.cap = cv2.VideoCapture(2)
        self.sound = AudioSegment.from_file('C:/Users/kingk/Desktop/人機介面表情包/mp提示音.wav', format='wav')
        
        # 設定 Mediapipe Holistic 模型的參數
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5
        )
        
        # 開始攝影機畫面的更新迴圈
        self.update_video()
    
    def update_video(self):
        # 讀取攝影機畫面
        success, image = self.cap.read()
        if success:
            # 將圖像轉換為 RGB 格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 進行人體姿勢檢測和距離檢測
            results = self.holistic.process(image)
            if check_distance(results.face_landmarks):
                # 如果距離太遠，播放提示音
                play(self.sound)
                self.status_label.setText("偵測有人")
            else:
                self.status_label.setText("無")
            
            # 在攝影機畫面上顯示人體姿勢和臉部特徵點
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            
            # 將圖像轉換為 QImage 格式，並顯示在 QLabel 元件上
            h, w, c = image.shape
            qimage = QImage(image.data, w, h, c*w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimage))
        
        # 設定 20 毫秒後再次更新攝影機畫面
        QTimer.singleShot(100, self.update_video)


def distance_between_points(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def check_distance(face_landmarks):
    if not face_landmarks:
        return False

    left_eye = face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE]
    right_eye = face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE]

    distance = distance_between_points(left_eye, right_eye)
    threshold = 0.02 # 兩眼距離（設定為 2 公尺）

    if distance > threshold:
        return True
    else:
        return False

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
