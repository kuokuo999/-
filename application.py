import tensorflow as tf
from tensorflow import keras
import librosa #https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import librosa.effects as le
import threading

# 加載模型
model_path = 'C:/Users/User/Desktop/speech/model/model2.h5'  
model = keras.models.load_model(model_path)

Original_path="C:/Users/User/Desktop/speech/voice_data_final"
path = os.listdir(Original_path)
print(path)



#應用部分(麥克風錄音)
import pyaudio
import wave
from playsound import playsound

# 設定錄音參數
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# 設定錄音檔案名稱和存儲路徑
WAVE_OUTPUT_FILENAME = "路人錄音檔.wav"
SAVE_PATH = "C:/Users/User/Desktop/speech/路人錄音檔/"


# 設定錄音前後提示音檔案名稱和存儲路徑
PROMPT_SOUND_FILENAME1 = "錄音前.wav"
PROMPT_SOUND_FILENAME2 = "錄音後.wav"

PROMPT_SOUND_PATH = "C:/Users/User/Desktop/speech/answer/"

# 錄音前提示音
def play_prompt_sound1():
    playsound(PROMPT_SOUND_PATH + PROMPT_SOUND_FILENAME1)
    
# 錄音後提示音
def play_prompt_sound2():
    playsound(PROMPT_SOUND_PATH + PROMPT_SOUND_FILENAME2)

# 在開始錄音前撥放提示音
play_prompt_sound1()


# 初始化錄音器
audio = pyaudio.PyAudio()

# 開始錄音
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("請說話")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
 data = stream.read(CHUNK)
 frames.append(data)

print("錄音完成，正在進行辨識")
 # 在錄音完成後播放提示音
play_prompt_sound2() 


# 關閉錄音器
stream.stop_stream()
stream.close()
audio.terminate()

# 儲存錄音結果
wf = wave.open(SAVE_PATH + WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print("存檔在: ", SAVE_PATH + WAVE_OUTPUT_FILENAME)



#AI測試
data_name="C:/Users/User/Desktop/speech/路人錄音檔/路人錄音檔.wav"
np_dtype = tf.float32.as_numpy_dtype
min_level_db = -80
max_level_db = 0
y, sr = librosa.load(data_name)
y = librosa.effects.preemphasis(y)
# y = signal.filtfilt(b, a, y)
fmax_=sr/2
# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                fmax=fmax_)
S_dB = librosa.power_to_db(S, ref=np.max)
S_dB = np.clip((S_dB - min_level_db) / (max_level_db - min_level_db), 0, 1)

resized_spec = tf.expand_dims(S_dB, -1)  #增添維度
# Resize spectrogram to 2D shape for CNN input
img_height = 128
img_width = 128
input_signal = tf.image.resize(resized_spec, size=[img_height, img_width])
print(np.shape(input_signal))

#AI predict
test=np.expand_dims (input_signal,axis=0)
y_predict = np.argmax(model.predict(test), axis=1)
# Use NumPy datatype
fig, ax = plt.subplots()
x = np.array(resized_spec[:,:,0], dtype=np_dtype)
img = librosa.display.specshow(x, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=fmax_, ax=ax)
 # fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Predict class is '+path[y_predict[0]])
print("Predict class is {}".format(path[y_predict[0]]))

#喇叭播放答案
from pydub import AudioSegment
from pydub.playback import play
import random
# 假設 path 包含 10 個類別名稱
#path = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5',
#        'class_6', 'class_7', 'class_8', 'class_9', 'class_10']
# 假設 y_predict 是一個包含預測結果索引的列表
#y_predict = [7]  # 使用索引 6 表示 'class_7'
# 提取預測結果的索引
predicted_class_index = y_predict[0]
# 讀取對應的類別名稱
predicted_class_name = path[predicted_class_index]
# ... 接下來是 handle_class 和其他函數的定義
# 使用預測的類別名稱調用對應的處理函數

def handle_class(class_name):
    switch_case = {
        'acoustic': handle_class_1,
        'autocar': handle_class_2,
        'department_office': handle_class_3,
        'eat': handle_class_4,
        'joke': handle_class_5,
        'lidar': handle_class_6,
        'project': handle_class_7,
        'psd': handle_class_8,
        'sing': handle_class_9,
        'weather_problem': handle_class_10,
    }

    # 根據類別名稱執行對應的函數
    function_to_execute = switch_case.get(class_name)
    if function_to_execute:
        function_to_execute()
    else:
        print("Invalid class name")


# 每個類別的處理函數

#電聲
def handle_class_1():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/電聲.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)

 audio_files = ["C:/Users/User/Desktop/speech/answer/acoustic/sound1.wav",
                "C:/Users/User/Desktop/speech/answer/acoustic/sound2.wav",
                "C:/Users/User/Desktop/speech/answer/acoustic/sound3.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
    
    
    
#自走車
def handle_class_2():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/自走車.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/autocar/car1.wav",
                "C:/Users/User/Desktop/speech/answer/autocar/car2.wav",
                "C:/Users/User/Desktop/speech/answer/autocar/car3.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)



#系辦    
def handle_class_3():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/系辦.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/department_office/office1.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
 
 
 
#美食    
def handle_class_4():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/美食.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/eat/food1.wav", 
                "C:/Users/User/Desktop/speech/answer/eat/food2.wav", 
                "C:/Users/User/Desktop/speech/answer/eat/food3.wav", 
                "C:/Users/User/Desktop/speech/answer/eat/food4.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
 
 
 
#笑話    
def handle_class_5():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/笑話.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/joke/laugh1.wav",
                "C:/Users/User/Desktop/speech/answer/joke/laugh2.wav",
                "C:/Users/User/Desktop/speech/answer/joke/laugh3.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
 
 last_audio_file = "C:/Users/User/Desktop/speech/answer/哈哈.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(last_audio_file)
 play(audio)
 
 
 
 
#光達    
def handle_class_6():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/光達.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/lidar/light1.wav",
                "C:/Users/User/Desktop/speech/answer/lidar/light2.wav",
                "C:/Users/User/Desktop/speech/answer/lidar/light3.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
    
    
    
#專題    
def handle_class_7():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/專題.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/project/project1.wav",]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)



#精密    
def handle_class_8():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/精密.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/psd/precision1.wav",
                "C:/Users/User/Desktop/speech/answer/psd/precision2.wav",
                "C:/Users/User/Desktop/speech/answer/psd/precision3.wav"]
 file = random.choice(audio_files)
  # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
    
    
    
#唱歌    
def handle_class_9():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/唱歌.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/sing/song1.wav",
                "C:/Users/User/Desktop/speech/answer/sing/song2.wav",
                "C:/Users/User/Desktop/speech/answer/sing/song3.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
     
     
     
#天氣    
def handle_class_10():
 # 指定你想要先播放的音檔
 first_audio_file = "C:/Users/User/Desktop/speech/answer/天氣.wav"
    
 # 讀取並播放你指定的音檔
 audio = AudioSegment.from_wav(first_audio_file)
 play(audio)
 
 audio_files = ["C:/Users/User/Desktop/speech/answer/weather_problem/weather1.wav",
                "C:/Users/User/Desktop/speech/answer/weather_problem/weather2.wav",
                "C:/Users/User/Desktop/speech/answer/weather_problem/weather3.wav"]
 file = random.choice(audio_files)
 # 讀取音頻文件
 audio = AudioSegment.from_wav(file)
 # 使用 pydub 播放音頻文件
 play(audio)
    
# 使用預測的類別名稱調用對應的處理函數
handle_class(predicted_class_name)