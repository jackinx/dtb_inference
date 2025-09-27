# dtb_inference
Deep Trader Baekdoosan daily crawling and inference

sequential task for Stock Prediction
1. crawling daily OHLCV data
1crawler_parallel_fromList.py

2. Prepare FEDformer inference
2FEDformer_preprocess_with_index_v9.py

3. Actual inference with predicted date as a input
3prediction_integrated_PPO_prepare.py
