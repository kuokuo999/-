import os
import cv2
import mediapipe as mp
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import random
import threading
import tkinter as tk

video_files = ['C:/Users/User/Desktop/提示音&動畫/5.mp4',]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def distance_between_points(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def check_distance(face_landmarks):
    if not face_landmarks:
        return False

    left_eye = face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE]
    right_eye = face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE]

    distance = distance_between_points(left_eye, right_eye)
    threshold = 0.02

    if distance > threshold:
        return True
    else:
        return False

cap = cv2.VideoCapture(1)

sound = AudioSegment.from_file('C:/Users/User/Desktop/提示音&動畫/mp提示音.wav', format='wav')
other_sound = AudioSegment.from_file('C:/Users/User/Desktop/提示音&動畫/觸發提示音.wav', format='wav')
other_script = 'C:/Users/User/Desktop/語音模型+動畫新.py'

def display_image_with_duration(image_path, duration_seconds, animation_width, animation_height):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (animation_width, animation_height), interpolation=cv2.INTER_AREA)
    cv2.imshow('MediaPipe Holistic', img)
    cv2.waitKey(int(duration_seconds * 1000))
    cv2.destroyAllWindows()


def main_loop(trigger_count, max_trigger_count):
    consecutive_triggers = 0
    video = cv2.VideoCapture(video_files[0])

    with mp_holistic.Holistic(
        min_detection_confidence=0.2,
        min_tracking_confidence=0.5) as holistic:

        cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        animation_width = 3200
        animation_height = 2000

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            video_success, video_frame = video.read()
            if not video_success:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_success, video_frame = video.read()

            video_frame = cv2.resize(video_frame, (animation_width, animation_height), interpolation=cv2.INTER_AREA)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

            if check_distance(results.face_landmarks):
                print("偵測到人")
                play(sound)
                trigger_count += 1
                consecutive_triggers += 1

                if consecutive_triggers >= max_trigger_count:
                    threading.Thread(target=play, args=(other_sound,)).start()
                    image_path = 'C:/Users/User/Desktop/1.png'  #問題清單，不能用中文
                    display_image_with_duration(image_path, other_sound.duration_seconds, animation_width, animation_height)
                    
                    os.system(f'python {other_script}')
                    
                    return True
            else:
                print("無")
                consecutive_triggers = 0
            # 重置連續觸發計數器，以便重新開始計數

            camera_image_height, camera_image_width, _ = image.shape
            # 設定鏡頭畫面大小
            resized_camera_image = cv2.resize(image, (int(camera_image_width * 1), int(camera_image_height * 1)))
            video_frame[-resized_camera_image.shape[0]:, :resized_camera_image.shape[1]] = resized_camera_image

            cv2.imshow('MediaPipe Holistic', video_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        return False


trigger_count = 0
max_trigger_count = 5  # 連續叫五次才會出觸發

def run_other_script():
    global trigger_count, max_trigger_count
    root.iconify()  # 添加此行以在按下按鈕時最小化窗口
    while True:  # 加入一個無限循环
        result = main_loop(trigger_count, max_trigger_count)
        if not result:
            trigger_count = 0  # 重置觸發計數器，以便重新開始計數


# 創建主視窗
root = tk.Tk()

# 設定主視窗的標題
root.title("任務控制介面")

# 設置窗口全屏
root.attributes('-fullscreen', True)
root.configure(bg='gray')

# 創建按鈕函數
def button_click(button_number):
    print("你按下了第 " + str(button_number) + " 個按鈕")

# 獲取螢幕的寬度和高度
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 計算按鈕之間的間距
button_spacing = 35

# 創建一個 Frame 控制元件
center_frame = tk.Frame(root)
center_frame.configure(bg='gray')

# 創建 Label 元件，用於顯示文字
label = tk.Label(center_frame,
                 text="任務控制介面",
                 font=("Times New Roman", 50),
                 fg='white',
                 bg='gray')


# 創建三個目的地按鈕
button1 = tk.Button(center_frame,
                    text="工109實驗室",
                    command=lambda: button_click(1),
                    width=11,
                    height=3,
                    font=("Times New Roman", 25),
                    fg='black',
                    bd=20,
                    activebackground='#A3D1D1'
                    )
button1.configure(bg='#D0D0D0')

button2 = tk.Button(center_frame,
                    text="系辦工225",
                    command=lambda: button_click(2),
                    width=11,
                    height=3,
                    font=("Times New Roman", 25),
                    fg='black',
                    bd=20,
                    activebackground='#A3D1D1'
                    )
button2.configure(bg='#D0D0D0')

button5 = tk.Button(center_frame,
                    text="工515",
                    command=lambda: button_click(5),
                    width=11,
                    height=3,
                    font=("Times New Roman", 25),
                    fg='black',
                    bd=20,
                    activebackground='#D2A2CC'
                    )
button5.configure(bg='#D0D0D0')

# 將 Label 元件加入 Frame 控制元件
label.pack(side=tk.TOP, pady=50, anchor=tk.N)

# 將 Frame 控制元件置中
center_frame.place(relx=0.5, rely=0, anchor=tk.CENTER)

# 將三個按鈕加入 Frame 控制元件，依序排列
button1.pack(side=tk.TOP, pady=button_spacing)
button2.pack(side=tk.TOP, pady=button_spacing)
button5.pack(side=tk.TOP, pady=button_spacing)

# 將 Frame 控制元件置中
center_frame.place(relx=0.25, rely=0.5, anchor=tk.W)


# 在右半邊添加一個 Frame 控制元件
right_frame = tk.Frame(root)
right_frame.configure(bg='gray')

# 在右下角添加一個按鈕來運行另一個程式
other_script_button = tk.Button(right_frame,
                                text="語音互動介面",
                                command=run_other_script,
                                width=11,
                                height=3,
                                font=("Times New Roman", 25),
                                fg='black',
                                bd=20,
                                activebackground='#8080C0'
                                )
other_script_button.configure(bg='#D0D0D0')

# 將按鈕加入右半邊 Frame 控制元件
other_script_button.pack(pady=button_spacing)

# 將右半邊 Frame 控制元件置中
right_frame.place(relx=0.75, rely=0.5, anchor=tk.N)

# 啟動主視窗的事件循環
root.mainloop()
                