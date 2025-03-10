import os
import numpy as np
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import seaborn as sns

# 指定數據庫路徑
database_path = "C:/Users/User/Desktop/speech/voice_data_final"

# 定義一些常量
mel_precision = 128
min_level_db = -80
max_level_db = 0

# 讀取和整理數據
x_data = []
y_data = []

for i, class_folder in enumerate(os.listdir(database_path)):
    class_folder_path = os.path.join(database_path, class_folder)
    
    if os.path.isdir(class_folder_path):
        for filename in os.listdir(class_folder_path):
            if filename.endswith('.wav'): # 或者其他音頻格式
                filepath = os.path.join(class_folder_path, filename)
                
                # 讀取音頻文件
                y, sr = librosa.load(filepath, sr=None)
                fmax_ = sr / 2
                
                # 進行高通濾波
                nyquist_freq = 0.5 * sr
                cutoff_freq = 100.0 / nyquist_freq
                b, a = signal.butter(4, cutoff_freq, btype='highpass')
                y = signal.filtfilt(b, a, y)

                # 提取Mel頻譜特徵
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_precision, fmax=fmax_)
                S_dB = librosa.power_to_db(S, ref=np.max)
                S_dB = np.clip((S_dB - min_level_db) / (max_level_db - min_level_db), 0, 1)
                resized_spec = tf.expand_dims(S_dB, -1)  
                resized_spec = tf.image.resize(resized_spec, size=[mel_precision, mel_precision])

                # 添加數據到列表
                x_data.append(resized_spec.numpy())  
                y_data.append(i)  

# 將資料集切割成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    np.array(x_data), np.array(y_data), test_size=0.25, random_state=42)

# 轉換標籤為 one-hot 編碼
num_classes = len(set(y_data)) # 自動獲得類別數量
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 定義模型輸入的形狀
input_shape = (mel_precision, mel_precision, 1)

# 創建模型
model = keras.Sequential([
    Flatten(input_shape=(128, 128, 1)),
    layers.Dense(128, activation="relu", input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(10, activation='softmax'),
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 顯示模型摘要
model.summary()

# 編譯模型
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 訓練模型
batch_size = 500
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=2500, validation_data=(X_test, y_test))



# 可視化訓練過程
# 繪製損失曲線
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# 繪製準確率曲線
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 評估模型
score = model.evaluate(X_test, y_test, verbose=0)
print(f"測試損失：{score[0]}")
print(f"測試準確率：{score[1]}")

# 畫混淆矩陣
# 對訓練數據進行預測
y_pred_train = model.predict(X_train)
y_pred_train = np.argmax(y_pred_train, axis=1)
y_true_train = np.argmax(y_train, axis=1)
cm_train = confusion_matrix(y_true_train, y_pred_train)
print("訓練數據的混淆矩陣:")
print(cm_train)

# 對測試數據進行預測
y_pred_test = model.predict(X_test)
y_pred_test = np.argmax(y_pred_test, axis=1)
y_true_test = np.argmax(y_test, axis=1)
cm_test = confusion_matrix(y_true_test, y_pred_test)
print("測試數據的混淆矩陣:")
print(cm_test)


# 画训练数据的混淆矩阵
plt.figure(figsize=(10,7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Training Data')
plt.show()

# 画测试数据的混淆矩阵
plt.figure(figsize=(10,7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Testing Data')
plt.show()