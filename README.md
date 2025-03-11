# 專題概述

大三下學期加入逢甲精密系主任領銜指導的跨領域專題研究，題目為「應用AI影像識別及光達輔助無磁軌AGV開發之研究—以逢甲大學工科院精密廊道與電梯場景為例」。透過智能化與標準化的方式進行重覆且繁瑣的測試動作，使其與相對應的設備產生配套方案，AGV系統為自動化生產線中一個不可或缺的環節，不受地理環境影響，可隨時裝卸貨物，解決耗費大量人力、時間和成本等問題，已是全球產業發展的共同趨勢。

綜觀現有AGV系統，多數皆透過鋪設色帶或磁軌來達成行走的動作，此方法維護成本相當高昂，且使用環境受到絕對限制。目前各廠商皆陸續投入無磁軌式AGV系統的開發，透過光學雷達感測反射光來遙測相對的距離，實現搬運車定位和運作等動作；然而，利用反射光可能導致許多問題的產生，如光線不足或光線變動大、雷射光束有穿透玻璃特性射程遠近等問題，皆會導致系統運作時，產生不可預期的錯誤和迷航。此研究以逢甲大學工程與科學學院廊道與電梯為主要場景，進行校園日常文件派送等動作的實作。利用AGV搭載LiDAR進行測距及定位等動作，並架設攝影機以影像識別來輔助LiDAR進行定位，確保其不會在特定環境迷航中斷或妨礙行走任務的執行。

除了開發AGV的自動導航外，許多關於AGV的研究項目也同時進行，如車架改造、人機介面設計及附屬於介面的語音互動功能。而我在專題研究上主要負責人機介面的設計、語音模型的設計及AGV的自動導航，並還協助了AGV結構設計與建立ROS系統。


<figure>
    <img src="https://github.com/user-attachments/assets/ae0247b2-3e2c-46fe-8f27-d3a4ebe0d37e" width="40%">
    <figcaption>流程規劃</figcaption>
</figure>
<img src="https://github.com/user-attachments/assets/a14cb563-c807-46b0-9fd6-1b502c0c787e" width="30%">


# CNN語音辨識模型

模型設計基於CNN的語音辨識架構，使系統具有高度適應性，能識別多種複雜的聲音信號。搭配親和力高的人機互動介面，使用者只須透過語音指令進行互動，實現優化的使用體驗。另為提高辨識準確性，模型訓練過程中，收集了系上學生提供約4000條聲音樣本，涵蓋多面向的問題和指令，如辦公室位置、系上活動公告、請求電腦唱歌或說笑話…等指令反覆訓練、深度學習，藉此提高辨識準確性。

數據預處理階段，將語音樣本進行標記與轉錄，以優化模型訓練。音頻樣本經過轉換、標籤分配與歸一化，確保訓練資料的質量和一致性。在聲音特徵提取中，將聲音信號轉化為數字格式，並採用梅爾頻率倒譜係數(Mel-Frequency Cepstral Coefficients；MFCC)擷取特徵，頻譜圖能夠截取聲音特徵，對於CNN具有良好訓練效果。

<figure>
    <img src="https://github.com/user-attachments/assets/0c53c144-675a-4896-b5ee-ae50e80da425" width="40%">
    <figcaption>語音流程圖</figcaption>
</figure>

<figure>
    <img src="https://github.com/user-attachments/assets/0a30141f-aabd-414f-bef7-d8a15fc7a6f0" width="30%">
    <figcaption>梅爾頻率倒譜係數</figcaption>
</figure>

<img src="https://github.com/user-attachments/assets/5f542ec0-2be9-40d2-baa7-c78cad9ddfd6" width="50%">
<img src="https://github.com/user-attachments/assets/00d2b706-37a4-405f-bcfa-c4335f04070f" width="49%">

實作過程中，利用深度神經網路多層提取及學習特徵的特性，再透過反向傳播算法來優化參數。經過2500個訓練周期(epochs)及超參數調整，模型達到92%的識別準確率，展現出優越的性能。

<img src="https://github.com/user-attachments/assets/c70c3438-555c-43cd-ae4d-99ad3e525002" width="70%">

圖中混淆矩陣可看出左圖有許多誤判使準確率下降，而右圖中誤判大幅減少，因此準確率較高。

# 人機介面&AGV無人搬運車改裝

人機介面的設計中，利用了Google的MediaPipe虹膜追蹤功能，在互動語音系統中捕捉互動者的眼睛位置，估算其與鏡頭的距離。當計算出的距離小於或等於兩公尺時，該系統將自動啟動語音互動功能。

這設計構想優勢在於透過使用MediaPipe技術，可以與標準鏡頭結合，從而達到與市面上專業產品類似的功能，降低成本。經過測試和精確的像素計算，發現在兩公尺的測距範圍內，所得的誤差極小且在可接受範圍之內。

綜上所述，MediaPipe的虹膜追蹤技術提供了一個既經濟又有效的策略，充分證明了開源工具在特定應用情境下的出色性能與潛在價值。




<figure>
    <img src="https://github.com/user-attachments/assets/34d31ed1-6a49-4d3d-8833-f5ab82c66766" width="43%">
    <figcaption>人機介面流程圖</figcaption>
</figure>

<figure>
    <img src="https://github.com/user-attachments/assets/a3de773e-32b2-4e82-b8c8-ac71842c4039" width="35%">
    <figcaption>任務控制介面</figcaption>
</figure>

<figure>
    <img src="https://github.com/user-attachments/assets/ca6cd3a5-39fc-4a1c-b54a-51db8ae93e87" width="40%">
    <figcaption>改裝前</figcaption>
</figure>
<figure>
    <img src="https://github.com/user-attachments/assets/303796fb-7f34-48c8-be5c-45051c692944" width="40%">
    <figcaption>改裝前</figcaption>
</figure>


<figure>
    <img src="https://github.com/user-attachments/assets/4464e386-985a-4823-85a7-b7130f6e346a" width="60%">
    <figcaption>改裝後細項</figcaption>
</figure>

<figure>
    <img src="https://github.com/user-attachments/assets/85ed961b-5af8-453c-a48e-1c6ad61eaceb" width="40%">
    <figcaption>改裝後</figcaption>
</figure>


# ROS（Robot Operating System )

<img src="https://github.com/user-attachments/assets/b309cca8-9f92-4efa-b76f-284db02de55d" width="50%">
<img src="https://github.com/user-attachments/assets/00ffed78-8226-4801-a48d-791d0aa86d85" width="40%">
<img src="https://github.com/user-attachments/assets/6cfddb7e-0e46-4a52-b460-0cc02db002be" width="40%">
<img src="https://github.com/user-attachments/assets/031b9b70-d41e-4211-9249-5bc2c7a3af00" width="55%">
<img src="https://github.com/user-attachments/assets/154bc430-0dec-4835-baf6-59b23d64c5ba" width="45%">
<img src="https://github.com/user-attachments/assets/6aadc797-0355-4987-98ef-9a17a92a5abf" width="45%">
<img src="https://github.com/user-attachments/assets/04215f9a-e7ab-4203-ae45-0e0a731dc420" width="50%">



