### Lab 6: DDPM - Conditional Synthetic Image Generation

📌任務目標: 實現一個條件式擴散模型 (Conditional DDPM)，根據 multi-label 條件生成合成影像。模型需理解輸入的物件標籤組合（如: 紅色的正方體、藍色的圓柱體等），並在正確的空間位置生成對應的影像內容

📊 Dataset:
- 使用 ICLEVR 數據集，包含 64x64 的影像與對應的物件標籤
- 條件: 共 24 種物件屬性組合，以多標籤（Multi-label）形式呈現

🛠️ 核心技術:
1. 多標籤條件輸入: 讓 UNet 聽懂指令
   - 做法: 在原始的 UNet2DModel 中加入一個 Embedding Layer（線性層），將 24 維的 One-hot 標籤向量轉換成與「時間嵌入（Time Embedding）」相同的維度（512 維）
   - 原因: 原始模型通常只支援單一類別。透過新增 Embedding 並將其與時間資訊「相加（Add）」，可以讓模型在去噪的每一個步驟中，同時考慮到「現在是第幾步」以及「我要畫的是什麼物件」

2. 雜訊排程（Noise Schedule）: 決定「破壞」的速度
   - 做法: 使用 DDPMScheduler 並選擇 Cosine Schedule (squaredcos_cap_v2)
   - 原因: 相比傳統的 Linear Schedule，Cosine Schedule 在加噪過程中資訊流失較為平滑，不會在初期就讓影像徹底變成雜訊
  
3. 去噪邏輯：從混亂中建立秩序
   - 做法: 訓練時，隨機加入雜訊並讓模型預測「被加進去的雜訊是什麼」；推理（Sampling）時，則從純高斯雜訊開始，讓模型重複 1000 步預測並扣除雜訊
   - 白話解決方案: 想像你在清理一張佈滿灰塵的照片。模型學會「辨識灰塵並把它撥掉」。只要模型知道什麼是「不該存在的灰塵」，最後剩下的就是清晰的目標影像
