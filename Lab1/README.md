### Lab 1: Neural Network with Back-propagation from Scratch

📌任務目標: 不依賴 PyTorch 或 TensorFlow 等現成框架，僅透過 NumPy 實作具備雙隱藏層的類神經網路。手寫出完整的訓練流程，並透過 Linear 與 XOR 問題驗證網路的收斂能力
- 從零實作: 建構包含前向傳播 (Forward Pass) 與權重更新 (Weight Update) 的完整流程
- 反向傳播: 應用 Chain Rule 手動推導並實作梯度計算
- 參數調校: 分析不同 learning rate、隱藏層單元數及 activation function 對 Loss 的影響

🛠️核心技術:
- 訓練流程實作: 遵循「餵入 Input → Forward 得到 Output → 計算 Loss → Backward 傳遞梯度 → Optimizer 更新權重」的標準流程
- 模組化架構:
  - Layer Class: 負責處理核心數學運算，包含該層的 forward、backward 與 update
  - Network Class: 作為容器封裝多個 Layer，依序呼叫各層功能。將架構拆分能避免在 Backpropagation 時因連鎖法則過於複雜而產生邏輯混亂
- 數學與優化引擎:
  - Activation function: Sigmoid、tanh、ReLU
  - Loss function: MSE
  - 實作 Learning Rate Scheduler 透過動態調整 lr 來優化收斂表現

📊實驗發現:
- 學習率影響: 較高的學習率能加速收斂，但過高可能導致 Loss 震盪或不收斂
- 非線性必要性: 對於 XOR 類型的非線性問題，移除 activation function 會導致模型無法有效學習分類邊界，Loss 遠高於具 activation function 的版本
- 模型容量: 增加隱藏層單元數量能提升模型對複雜模式的擬合能力，但需注意 Overfitting 的風險 
