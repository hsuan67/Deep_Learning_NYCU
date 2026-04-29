### Lab 2: EEG Classification with EEGNet and DeepConvNet

📌任務目標: 實作兩種經典的腦電波 (EEG) 分類模型: EEGNet 與 DeepConvNet。透過嘗試三種 activation function（ReLU, Leaky ReLU, ELU），分析不同架構與函數對二元分類準確度的影響，並挑戰在測試集上達到高準確率

📊Dataset: 資料來源包含 S4b 與 X11b 兩個檔案，經處理後形成 1080 筆、維度為 (1080, 1, 2, 750) 的資料
- 1080: 總資料筆數
- 1 (Channel): 為了符合 PyTorch CNN 對三維資料（Channel, H, W）的預算需求，手動擴增的維度
- 2 (Electrode Channels):電波圖的兩個通道，對應到 CNN 運算中的「寬度」
- 750 (Time points): 電波圖的時間軸長度，對應到 CNN 運算中的「長度」
綜上所述，資料其實是總共有 1080 筆的電波圖，每個電波圖都有兩個 channel 並時間長 750，最後再擴增出一維輔助計算

💡實驗發現:
- Depthwise Convolution: 透過將 Group 數設為 Input Channel 數，強制模型在該層不進行 Channel 間的交互運算，大幅度精簡了參數量並減少運算負擔，這在處理資源受限的電波資料時非常有效
- Batch Normalization: 透過對梯度進行 Normalization，有效控制了權重更新的尺度，防止梯度爆炸或消失，是讓深層模型（如 DeepConvNet）能順利訓練的功臣
- activation function: 實測發現 Leaky ReLU / ELU 確實解決了 Dying ReLU 的問題。透過在負半區保留微小梯度，避免了神經元死掉後無法更新權重的窘境
