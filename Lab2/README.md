### Lab 2: EEG Classification with EEGNet and DeepConvNet

📌任務目標: 實作兩種經典的腦電波 (EEG) 分類模型: EEGNet 與 DeepConvNet。透過嘗試三種 activation function（ReLU, Leaky ReLU, ELU），分析不同架構與函數對二元分類準確度的影響，並挑戰在測試集上達到高準確率

📊Dataset: 資料來源包含 S4b 與 X11b 兩個檔案，經處理後形成 1080 筆、維度為 (1080, 1, 2, 750) 的資料
- 1080: 總資料筆數
- 1 (Channel): 為了符合 PyTorch CNN 對三維資料（Channel, H, W）的預算需求，手動擴增的維度
- 2 (Electrode Channels): 電波圖的兩個通道，對應到 CNN 運算中的「寬度」
- 750 (Time points): 電波圖的時間軸長度，對應到 CNN 運算中的「長度」
綜上所述，資料其實是總共有 1080 筆的電波圖，每個電波圖都有兩個 channel 並時間長 750，最後再擴增出一維輔助計算

🛠️核心技術:
- EEGNet: 核心特色是使用 Depthwise Separable Convolution。先提取時間特徵，再提取空間（通道間）特徵，最後混合資訊，這能大幅減少參數量
  1. 時間特徵提取 (Temporal Convolution)
     - 腦電波訊號是非常快速的序列變化。這個步驟就像是在聽一段音樂，長條卷積核 (kernel size 為 (1, 51)) 負責掃描「一段時間」內的波形變化，提取出低階的時間特徵
  2. 空間特徵提取 (Depthwise Convolution)
     - 分組運算: groups = 16，這 16 個輸入通道（從上一步產生的）分別獨立進行卷積，沒有 Channel 間的交互資訊，節省運算量
     - 空間混合: 模型會去比較「頭部不同位置」的電極訊號差異（例如左腦與右腦的信號對比）
     - why?
       - 減少參數量: 如果用一般的卷積（全部通道一起算），參數量會爆炸。分組計算可以讓模型在不犧牲性能的前提下，大幅精簡大小
       - 提取空間模式: 腦電波在不同大腦區域的分布具有特定意義，這一步專門負責捕捉「空間位置」的資訊
  3. 特徵混合與分類 (Separable Convolution)
     - 使用 1 x 15 的卷積核進行 Pointwise Convolution
     - 整合前面提取到的時間與空間資訊 。將不同通道間的資訊重新混合，找出最終能代表這筆 EEG 資料的特徵點，最後再透過 Flatten 轉成一維向量進行分類

- 訓練時需先將 model.train() 開啟（以確保 Dropout 與 BatchNorm 正常運作），並在每一筆 batch 運算前將梯度清零（optimizer.zero_grad()）
- 測試時呼叫 model.eval() 關閉 Dropout，並使用 torch.no_grad() 確保不計算梯度，節省記憶體
- Optimizer & Scheduler: 使用 AdamW 優化器與 CosineAnnealingWarmRestarts 調度器。後者會讓學習率隨 epoch 呈週期性變動，有助於跳出局部優化點

💡實驗發現:
- Depthwise Convolution: 透過將 Group 數設為 Input Channel 數，強制模型在該層不進行 Channel 間的交互運算，大幅度精簡了參數量並減少運算負擔，這在處理資源受限的電波資料時非常有效
- Batch Normalization: 對每個 mini batch 做標準化，大的變小、小的變大， 讓它們平均一點， 防止梯度消失或梯度爆炸（每一層的輸出都有 B atchNorm ，確保輸入數據在每一層都保持適當的尺度，可以加快收斂速度）
- Dropout: 防止過度依賴某個 neuron ，減少 overfitting 的風險（在每個迭代以一定的機率丟棄 hidden layer 的神經元，而被丟棄的神經元不會傳遞訊息。在反向傳播時，被丟棄的神經元梯度是 0，這樣模型在訓練時就不會過度依賴某個神經元）
- Activation function: 實測發現 Leaky ReLU / ELU 確實解決了 Dying ReLU 的問題。透過在負半區保留微小梯度，避免了神經元死掉後無法更新權重的窘境
- Batch 的目的是想算出一部分資料的 loss 和 gradient 就去更新模型參數，不用每次都拿全部資料來計算
- 當 batch size 很小時 ，模型看了一點資料就更新一次，可能造成學習的方向變化很大不好收斂；當 batch size 調大時，模型的更新方向會更接近資料的全域情形。模型會一次看過更多的資料再去計算梯度決定更新方向。因此模型的學習效果通常會更好
