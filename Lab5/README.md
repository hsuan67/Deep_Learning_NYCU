### Lab 5: Reinforcement Learning - DQN & DDPG

📌 任務目標: 透過 訓練 Reinforcement Learning 訓練 AI 解決不同類型的決策問題:
1. 離散動作控制: 在月球太空船（LunarLander-v2）中選擇四種固定動作之一（左、右、上、不噴射）
2. 連續動作控制: 在連續版月球太空船（LunarLanderContinuous-v2）中精確決定引擎噴射的力道(-1 ~ 1 之間)
3. 影像視覺控制: 在 Atari 經典遊戲（Breakout）中，AI 需直接觀察螢幕像素來學習打磚塊

📊 Dataset / Environment
- LunarLander: 輸入是太空船的位置、速度等 8 個數值
- Breakout: 輸入是 84x84 的灰階影像
- 測試指標: 每個任務測試 10 回合，計算平均獎勵，驗證模型是否能在不同隨機種子下穩定過關

🛠️ 核心技術
1. Experience Replay
   -  目的: 打破資料的時間相關性，讓模型不會「只記得剛才發生的事」
   -  做法: 建立一個 ReplayMemory（緩衝區），將過往的 (state, action, reward, next state, done) 存入。訓練時從中「隨機抽樣」出一個 Batch
   -  AI 的經驗是連續的（現在的畫面跟上一秒很像）。如果照順序學，網路會因為資料太像而產生偏見。隨機抽樣就像是「複習考古題」，讓 AI 能同時從不同時間點的經驗中學習，確保訓練穩定

2. Deep Q-Network (DQN) —— 適用於離散空間
   - Target Network Mechanism (目標網路機制):
     - 目的: 解決訓練時目標不斷變動（Moving Target）的問題
       - 如果只用一套網路，預測值跟目標值會一起動，像獵人追逐會隨自己腳步移動的靶心。固定 Target Network 可以讓獵人有穩定的標靶可以瞄準
     - 做法: 實作兩套參數結構相同的網路
       - Behavior Network: 負責跟環境互動、選動作，並且每一小步都會更新參數來學習
       - Target Network: 負責產出計算標準。它不會每一步都更新，而是每隔一段時間才從 Behavior Network 複製一次參數過來
   - ϵ-greedy 策略:
     - 目的: 平衡「探索新的可能」與「利用學到最強招式）」
       - 純 Greedy 選擇會讓 AI 變得很固執，永遠只做它目前認為對的事。加入 ϵ 強迫它去「試錯」，通常能讓 AI 發現原本沒想到的高分路徑
     - 做法: 隨機選擇動作的機率會隨著時間減少

3. Deep Deterministic Policy Gradient (DDPG) —— 適用於連續空間
   - Actor-Critic 架構
     -  Actor: 負責看著狀態 s，直接輸出一個確定的動作力道值（如噴射力道 0.75）
     -  Critic: 評估「在這個狀態下，做這個動作的價值是多少」
   - Soft Update
      - 目的: 讓連續動作的目標變化極度平滑，防止訓練崩潰
      - 做法: 每次訓練後，只把 Behavior Net 的一小部分挪給 Target Net
      - 相對於 DQN 的「硬拷貝」，軟更新讓 Target 網路以「溫和挪移」的方式進化，這對對參數變動極其敏感的連續控制（DDPG）來說很重要

4. Atari Breakout 特殊技巧 (影像處理)
   - Frame Stacking (疊加幀)
     - 目的: 讓 AI 具備「時間感」
     - 做法: 將連續 4 幀影像疊在一起作為輸入
     - 單張照片看不出球是往哪動。透過疊加，AI 才能理解「速度」與「方向」
   - Episodic Life (生命敏感度)
     - 目的：強迫 AI 學會「珍惜生命」
     - 做法：只要 AI 掉了一顆球（少掉一條命），就視為一個回合結束
     - 這能讓 AI 對「死亡」產生恐懼，進而學習更強的防禦策略
   - Fire (開火起手式)
     - 目的: 避免 AI 在遊戲開始前原地發呆
     - 做法: 在遊戲重置後強制觸發「Fire」動作
     - Atari 遊戲常需按開火鍵才會發球，這確保 AI 每一回合都在真正的訓練狀態
  
