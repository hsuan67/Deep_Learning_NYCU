### Lab 4: Conditional VAE for Video Prediction

📌任務目標: 透過輸入一段人體姿勢序列（Pose sequence）以及第一幀起始圖像，模型需預測後續完整的動作影片
- 影像生成: 學習從潛在空間（Latent Space）重構高維影像資訊
- 序列預測: 結合姿勢資訊，讓模型在長序列下仍能維持影像生成的穩定性
- 優化技巧: 實作 Reparameterization trick、Teacher Forcing 策略與 KL Annealing 以提升模型表現

📊Dataset:
- 內容: 包含訓練影片（影像幀）與對應的人體骨架圖（Pose labels）
- 訓練與測試差異: 訓練影片長度固定為 16 幀，但測試影片則長達 630 幀
- 維度處理: 為了配合 frame-by-frame 處理，訓練時會將 Tensor 維度從 (B, seq, C, H, W) 重排為 (seq, B, C, H, W)

🛠️核心技術:
1. 訓練流程與影像生成
  模型透過 Feature Extraction 提取首幀特徵作為參考 ，隨後每一幀依序透過以下流程:
  - Posterior Prediction: 將預測的人體特徵與 Label 特徵傳入 Gaussian Predictor 得到潛在變數 z 的均值與方差
  - Decoder Fusion：融合人體特徵、姿勢特徵與隨機採樣的 z
  - Generator：根據融合後的參數生成下一幀影像

2. 重參數化技巧 (Reparameterization Trick)
  為了解決隨機採樣無法反向傳播的問題，我們將隨機性轉移到雜訊 ϵ 上
  - 計算標準差
  - 生成隨機雜訊
  - 計算潛在變數

3. Teacher Forcing 策略
  為了防止模型在訓練初期「一步錯、步步錯」
  - 機制: 模型決定使用 Ground Truth 的特徵或是上一步預測的輸出作為當前輸入
  - 更新策略: 初始 Ratio 為 1.0，從第 10 個 Epoch 開始遞減，引導模型逐漸學會獨立預測

4. KL Annealing
  為了平衡 Reconstruction Loss (MSE) 與 KL Divergence，實作了兩種權重更新策略:
  - Monotonic: 權重 β 每步增加 0.1 直到 1.0，過程較穩定 
  - Cyclical: 權重在週期內線性變化，有助於模型交替專注於數據重建與潛在空間探索

💡實驗發現:
- 長影片預測問題：訓練僅用 16 幀，但測試需預測 630 幀。若模型過度依賴 Teacher Forcing（TF），預測長影片時容易崩潰 。因此，我選擇調低 TF 比例，強迫模型在訓練中學會減少依賴 Ground Truth
- 收斂性分析:
  - 無 Annealing: Loss 曲線會非常混亂且難以收斂
  - Monotonic: 收斂最平滑，能穩定地利用潛在空間
- KL Divergence 的意義：其本質是計算 z 分佈與標準高斯分佈間的機率差異，確保生成的影像具有一定的多樣性
