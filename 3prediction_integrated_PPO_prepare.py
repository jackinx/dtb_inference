# -*- coding: utf-8 -*-
"""
FEDformer PPO 데이터 생성 및 예측 파이프라인 (v22.1 - 버그 수정)

주요 기능 (v22 대비 개선):
1.  [버그 수정] pandas Series에서 마지막 날짜를 가져올 때 발생하던 KeyError를 .iloc[-1]을 사용하여 해결.
2.  [안정성 강화] 필터링된 날짜 Series에서도 .iloc를 사용하여 안정적인 인덱싱 보장.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sqlite3
import gc
import json
import joblib
import copy
from pathlib import Path
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error
from tqdm import tqdm
from datetime import datetime, timedelta
import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional
from contextlib import contextmanager
import matplotlib
from matplotlib import font_manager as fm # <--- font_manager import
matplotlib.use("Agg")
import matplotlib.pyplot as plt




# ==================================================
# 1. 경로 및 모듈 설정
# ==================================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

FEDFORMER_ROOT = PROJECT_ROOT / 'FEDformer-master'
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
if str(FEDFORMER_ROOT) not in sys.path: sys.path.insert(0, str(FEDFORMER_ROOT))
try:
    from FEDformer import Model as FEDformer_base
    from utils.timefeatures import time_features
except ImportError as e:
    sys.exit(f"❌ CRITICAL ERROR: FEDformer 모듈 임포트 실패: {e}")

# ==================================================
# 2. 모델 클래스 정의
# ==================================================
class FEDformerWithEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.ticker_embedding = nn.Embedding(self.config.num_classes, self.config.embedding_dim)
        self.embedding_dropout = nn.Dropout(self.config.embedding_dropout)
        config_for_base = copy.deepcopy(self.config)
        config_for_base.enc_in += self.config.embedding_dim
        config_for_base.dec_in += self.config.embedding_dim
        self.base_model = FEDformer_base(config_for_base)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ticker_ids):
        emb = self.ticker_embedding(ticker_ids.squeeze(-1))
        emb = self.embedding_dropout(emb)
        emb_enc = emb.unsqueeze(1).repeat(1, self.config.seq_len, 1)
        emb_dec = emb.unsqueeze(1).repeat(1, self.config.label_len + self.config.pred_len, 1)
        x_enc_with_emb = torch.cat([x_enc, emb_enc], dim=-1)
        x_dec_with_emb = torch.cat([x_dec, emb_dec], dim=-1)
        return self.base_model(x_enc_with_emb, x_mark_enc, x_dec_with_emb, x_mark_dec)

# ==================================================
# 3. 설정 클래스 및 하이퍼파라미터
# ==================================================
class Config:
    def __init__(self):
        self.PROJECT_ROOT = PROJECT_ROOT
        self.V13_RESULT_DIR = self.PROJECT_ROOT / "training_results_v13"
        self.V14_RESULT_DIR = self.PROJECT_ROOT / "training_results_v14_retrain"
        self.PROCESSED_DATA_DIR = self.PROJECT_ROOT / "processed_data_v9"
        
        self.PROCESSED_DB_PATH = self.PROCESSED_DATA_DIR / "processed_stock_data.db"
        self.METADATA_DIR = self.PROCESSED_DATA_DIR / "metadata"
        self.SCALER_DIR = self.PROCESSED_DATA_DIR / "scalers"
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 전체 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 1. 실행 시점 기준의 고유한 타임스탬프 생성
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 2. PPO 예측 결과 상위 폴더
        predictions_base_dir = self.PROJECT_ROOT / "predictions_v22_ppo"
        
        # 3. 타임스탬프를 이름으로 하는 실행 결과 폴더 생성
        self.RUN_OUTPUT_DIR = predictions_base_dir / run_timestamp
        self.RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 4. 각 결과물의 경로를 새로운 폴더 기준으로 재설정
        self.BACKTEST_PLOTS_DIR = self.RUN_OUTPUT_DIR / "backtest_plots"
        self.BACKTEST_PLOTS_DIR.mkdir(exist_ok=True)
        
        # 엑셀 파일은 폴더명에 타임스탬프가 있으므로 파일명은 단순하게 변경
        self.RECOMMENDATION_EXCEL_PATH = self.RUN_OUTPUT_DIR / "final_recommendations.xlsx"
        
        # 참고: ppo_predictions.db는 모든 실행 결과를 누적하는 용도이므로
        # 타임스탬프 폴더 밖, 상위 폴더에 위치시키는 것이 더 적합합니다.
        self.PPO_DB_PATH = predictions_base_dir / "ppo_predictions.db"
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 전체 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        
        source_project_root = self.PROJECT_ROOT.parent / "DeepTrader_baekdoosan"
        if not source_project_root.exists():
            source_project_root = Path(r"C:\Users\jacki\OneDrive\Documents\anaconda_projects\DeepTrader_baekdoosan")
        self.SOURCE_DB_PATH = source_project_root / "stock_data.db"
        
        self.MAX_CONCURRENT_PROCESSES = max(1, mp.cpu_count() // 2)
        self.DB_TIMEOUT = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.RECOMMEND_COUNT_BULL = 15
        self.RECOMMEND_COUNT_NEUTRAL = 7
        self.RECOMMEND_COUNT_BEAR = 3
        self.MARKET_UP_THRESHOLD = 0.2
        self.MARKET_DOWN_THRESHOLD = -0.3
        
        self.MIN_AVG_VOLUME = 100000
        self.MAX_PER_SECTOR = 3

MODEL_CONFIGS = {
    'base': SimpleNamespace(config_name='base', version='Fourier', model_name='FEDformer', seq_len=60, label_len=30, pred_len=1, moving_avg=25, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff_multiplier=4, dropout=0.1, activation='gelu', output_attention=False, modes=64, mode_select='random', L=3, base='Fourier', cross_activation='tanh', embed='timeF', freq='d'),
    'tuning_v1': SimpleNamespace(config_name='tuning_v1', version='Fourier', model_name='FEDformer', seq_len=60, label_len=30, pred_len=1, moving_avg=25, d_model=256, n_heads=8, e_layers=2, d_layers=1, d_ff_multiplier=4, dropout=0.2, activation='gelu', output_attention=False, modes=64, mode_select='random', L=3, base='Fourier', cross_activation='tanh', embed='timeF', freq='d'),
    'shallow_wide': SimpleNamespace(config_name='shallow_wide', version='Fourier', model_name='FEDformer', seq_len=60, label_len=30, pred_len=1, moving_avg=25, d_model=384, n_heads=12, e_layers=1, d_layers=1, d_ff_multiplier=3, dropout=0.35, activation='gelu', output_attention=False, modes=48, mode_select='random', L=3, base='Fourier', cross_activation='tanh', embed='timeF', freq='d'),
    'deep_narrow': SimpleNamespace(config_name='deep_narrow', version='Fourier', model_name='FEDformer', seq_len=60, label_len=30, pred_len=1, moving_avg=25, d_model=256, n_heads=8, e_layers=4, d_layers=2, d_ff_multiplier=4, dropout=0.2, activation='gelu', output_attention=False, modes=64, mode_select='random', L=4, base='Fourier', cross_activation='tanh', embed='timeF', freq='d'),
    'high_freq': SimpleNamespace(config_name='high_freq', version='Fourier', model_name='FEDformer', seq_len=40, label_len=20, pred_len=1, moving_avg=15, d_model=320, n_heads=10, e_layers=2, d_layers=1, d_ff_multiplier=4, dropout=0.25, activation='gelu', output_attention=False, modes=96, mode_select='random', L=3, base='Fourier', cross_activation='tanh', embed='timeF', freq='d'),
    'low_freq': SimpleNamespace(config_name='low_freq', version='Fourier', model_name='FEDformer', seq_len=80, label_len=40, pred_len=1, moving_avg=35, d_model=384, n_heads=8, e_layers=2, d_layers=1, d_ff_multiplier=4, dropout=0.15, activation='gelu', output_attention=False, modes=32, mode_select='low', L=3, base='Fourier', cross_activation='tanh', embed='timeF', freq='d'),
    'regularized': SimpleNamespace(config_name='regularized', version='Fourier', model_name='FEDformer', seq_len=60, label_len=30, pred_len=1, moving_avg=25, d_model=192, n_heads=6, e_layers=2, d_layers=1, d_ff_multiplier=2, dropout=0.4, activation='gelu', output_attention=False, modes=40, mode_select='random', L=3, base='Fourier', cross_activation='tanh', embed='timeF', freq='d')
}
for cfg_item in MODEL_CONFIGS.values():
    cfg_item.d_ff = int(cfg_item.d_model * cfg_item.d_ff_multiplier)
    cfg_item.embedding_dim=16; cfg_item.embedding_dropout=0.2

# ==================================================
# 4. 데이터 및 유틸리티
# ==================================================
@contextmanager
def suppress_stdout():
    """컨텍스트 내에서 발생하는 print 출력을 숨깁니다."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class DataHandler:
    def __init__(self, processed_db_path: Path, source_db_path: Path, timeout: int):
        if not processed_db_path.exists() or not source_db_path.exists():
            sys.exit(f"❌ DB 파일 오류: DB 경로를 확인하세요.\n- {processed_db_path}\n- {source_db_path}")
        self.processed_db_path = processed_db_path
        self.source_db_path = source_db_path
        self.timeout = timeout
    def get_stock_metadata(self) -> Dict[str, Dict[str, str]]:
        with sqlite3.connect(self.source_db_path, timeout=self.timeout) as conn:
            try:
                df = pd.read_sql_query("SELECT ticker, name, market, sector FROM stocks", conn)
                df['ticker'] = df['ticker'].astype(str) # <--- 이 줄을 추가해주세요.
            except (pd.errors.DatabaseError, sqlite3.OperationalError) as e:
                if 'no such column: sector' in str(e):
                    print("  - 경고: 'stocks' 테이블에 'sector' 컬럼이 없습니다. 섹터 분산 기능이 비활성화됩니다.")
                    df = pd.read_sql_query("SELECT ticker, name, market FROM stocks", conn)
                else: raise e
        return df.set_index('ticker').to_dict('index')
    def get_all_trading_dates(self) -> pd.Series:
        """[수정] 반환 타입을 DatetimeIndex에서 Series로 명시"""
        with sqlite3.connect(self.processed_db_path, timeout=self.timeout) as conn:
            df = pd.read_sql_query("SELECT DISTINCT date FROM processed_daily_prices ORDER BY date", conn)
        return pd.to_datetime(df['date'])
    # 6prediction_integrated_PPO_prepare.py 파일의 DataHandler 클래스

    def get_price_data_until(self, ticker: str, end_date: str, num_rows: int) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.processed_db_path, timeout=self.timeout) as conn:
            # [수정] date 컬럼을 명시적으로 date 타입으로 변환하여 비교
            query = f"SELECT * FROM processed_daily_prices WHERE ticker = ? AND date(date) <= ? ORDER BY date DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(ticker, end_date, num_rows))
        if df.empty or len(df) < num_rows: return None
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date', ascending=True).reset_index(drop=True)
        
        
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분을 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    def get_avg_volume(self, ticker: str, days: int = 20) -> float:
        """지정된 종목의 최근 N일 평균 거래량을 원본 DB에서 조회합니다."""
        # 변환되지 않은 원본 거래량을 가져오기 위해 source_db_path를 사용합니다.
        with sqlite3.connect(self.source_db_path, timeout=self.timeout) as conn:
            try:
                # 해당 종목의 데이터를 날짜 내림차순으로 정렬하여 최근 `days`개 만큼 조회
                query = f"SELECT volume FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?"
                df = pd.read_sql_query(query, conn, params=(ticker, days))
                
                # 데이터가 존재하면 거래량의 평균을 계산하고, 없으면 0을 반환합니다.
                if not df.empty:
                    return float(df['volume'].mean())
                else:
                    return 0.0
            except Exception as e:
                # 혹시 모를 오류 발생 시 경고를 출력하고 0을 반환합니다.
                print(f"Warning: Could not get avg volume for {ticker}. Error: {e}")
                return 0.0
                

def get_next_trading_day(last_date: datetime, all_trading_dates: pd.Series) -> datetime:
    future_dates = all_trading_dates[all_trading_dates > last_date]
    return future_dates.iloc[0] if not future_dates.empty else last_date + pd.tseries.offsets.BDay(1)

def inverse_transform_price(scaler: StandardScaler, scaled_value: float, close_idx: int) -> float:
    dummy_array = np.zeros((1, scaler.n_features_in_))
    dummy_array[0, close_idx] = scaled_value
    unscaled_log_price = scaler.inverse_transform(dummy_array)[0, close_idx]
    return np.expm1(unscaled_log_price)

# ==================================================
# 5. 예측 실행 로직
# ==================================================
def run_inference(ticker: str, model_info: Dict, df_recent: pd.DataFrame, next_trading_day: datetime, cfg: Config) -> Optional[float]:
    try:
        config_name = model_info['config_name']
        model_path = model_info['model_path']
        if not model_path.exists(): return None
        
        with suppress_stdout():
            model_config = copy.deepcopy(MODEL_CONFIGS[config_name])
            with open(cfg.METADATA_DIR / f"{ticker}.json", 'r') as f: features = json.load(f)['features']
            model_config.enc_in = model_config.dec_in = len(features)
            model_config.c_out = 1
            model = FEDformer_base(model_config).to(cfg.device)
            model.load_state_dict(torch.load(model_path, map_location=cfg.device, weights_only=False)['model_state_dict'])
            model.eval()

        scaler = joblib.load(cfg.SCALER_DIR / f"{ticker}.pkl")
        data_to_scale = df_recent[features].astype(np.float32)
        scaled_data = scaler.transform(data_to_scale)
        time_marks = time_features(pd.DatetimeIndex(df_recent['date']), freq=model_config.freq).transpose()
        x_enc = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        x_mark_enc = torch.tensor(time_marks, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        dec_inp_context = x_enc[:, -model_config.label_len:, :]
        dec_inp_zeros = torch.zeros(1, model_config.pred_len, model_config.dec_in, device=cfg.device)
        x_dec = torch.cat([dec_inp_context, dec_inp_zeros], dim=1)
        decoder_dates = pd.DatetimeIndex(df_recent['date'].iloc[-model_config.label_len:].tolist() + [next_trading_day])
        x_mark_dec = torch.tensor(time_features(decoder_dates, freq=model_config.freq).T, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            pred_scaled = model(x_enc, x_mark_enc, x_dec, x_mark_dec)[0, -1, 0].item()
        close_idx = features.index('close')
        return inverse_transform_price(scaler, pred_scaled, close_idx)
    except Exception:
        return None
# 6prediction_integrated_PPO_prepare.py 파일

# 이 함수 전체를 아래 내용으로 교체해주세요.
def run_backtest_and_visualize(
    ticker: str, 
    ticker_name: str, 
    model_info: Dict, 
    all_trading_dates: pd.Series, 
    cfg: Config, 
    next_day_prediction: float, # <--- 이 인자가 누락되었을 가능성이 높습니다.
    backtest_days: int = 20
) -> Dict:
    """추천된 종목의 과거 N일간 예측 + 미래 1일 예측을 시각화합니다."""
    
    data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
    model_config = MODEL_CONFIGS[model_info['config_name']]
    
    end_date = all_trading_dates.iloc[-1]
    backtest_dates = all_trading_dates[all_trading_dates <= end_date].iloc[-backtest_days:]
    
    predictions, actuals, last_closes = [], [], []
    
    for date in backtest_dates:
        end_of_window = date - pd.Timedelta(days=1)
        df_recent = data_handler.get_price_data_until(ticker, end_of_window.strftime('%Y-%m-%d'), model_config.seq_len)
        
        if df_recent is None or len(df_recent) < model_config.seq_len:
            continue
            
        df_actual = data_handler.get_price_data_until(ticker, date.strftime('%Y-%m-%d'), 1)
        if df_actual is None or df_actual.empty:
            continue
        
        with open(cfg.METADATA_DIR / f"{ticker}.json") as f: features = json.load(f)['features']
        close_idx = features.index('close')
        
        predicted_price = run_inference(ticker, model_info, df_recent, date, cfg)
        if predicted_price is None:
            continue
        
        actual_price_log = df_actual.iloc[0][features].iloc[close_idx]
        last_close_log = df_recent.iloc[-1][features].iloc[close_idx]
        
        predictions.append(predicted_price)
        actuals.append(np.expm1(actual_price_log))
        last_closes.append(np.expm1(last_close_log))

    if not predictions:
        return {'backtest_f1': 0, 'backtest_nrmse': 1.0, 'plot_path': 'N/A'}
    
    y_true, y_pred, y_last = np.array(actuals), np.array(predictions), np.array(last_closes)
    true_dir, pred_dir = (y_true > y_last).astype(int), (y_pred > y_last).astype(int)
    f1 = f1_score(true_dir, pred_dir, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 1e-5 else 0

    try:
        last_backtest_date = backtest_dates.iloc[-1]
        next_pred_date = get_next_trading_day(last_backtest_date, all_trading_dates)
        plot_dates = backtest_dates[-len(y_pred):].tolist() + [next_pred_date]
        plot_predictions = y_pred.tolist() + [next_day_prediction]

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rc('font', family='Malgun Gothic' if sys.platform.startswith('win') else 'AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(backtest_dates[-len(y_true):], y_true, label='실제 주가 (Actual Price)', marker='o', markersize=4, linestyle='-')
        ax.plot(plot_dates, plot_predictions, label='예측 주가 (Predicted Price)', marker='x', markersize=4, linestyle='--')
        ax.set_title(f'최근 {backtest_days}일 백테스트: {ticker_name} ({ticker})', fontsize=16)
        ax.set_ylabel('종가 (KRW)', fontsize=12)
        ax.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = cfg.BACKTEST_PLOTS_DIR / f"{ticker}_backtest.png"
        plt.savefig(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"Plotting Error for {ticker}: {e}")
        plot_path = "N/A"

    return {'backtest_f1': round(f1, 4), 'backtest_nrmse': round(nrmse, 4), 'plot_path': str(plot_path)}

'''
def run_backtest_and_visualize(
    ticker: str, 
    ticker_name: str, 
    model_info: Dict, 
    all_trading_dates: pd.Series, 
    cfg: Config, 
    backtest_days: int = 20
) -> Dict:
    """추천된 종목의 과거 N일간 예측을 수행하고 성능 계산 및 시각화를 진행합니다."""
    
    data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
    model_config = MODEL_CONFIGS[model_info['config_name']]
    
    # 백테스팅 기간 설정
    end_date = all_trading_dates.iloc[-1]
    backtest_dates = all_trading_dates[all_trading_dates <= end_date].iloc[-backtest_days:]
    
    predictions, actuals, last_closes = [], [], []
    
    # 과거 N일에 대해 하루씩 예측 수행
    for date in backtest_dates:
        # 예측 대상일(date)의 전날까지 데이터를 가져옴
        end_of_window = date - pd.Timedelta(days=1)
        df_recent = data_handler.get_price_data_until(ticker, end_of_window.strftime('%Y-%m-%d'), model_config.seq_len)
        
        if df_recent is None or len(df_recent) < model_config.seq_len:
            continue
            
        # 실제 값(target) 가져오기
        df_actual = data_handler.get_price_data_until(ticker, date.strftime('%Y-%m-%d'), 1)
        if df_actual is None or df_actual.empty:
            continue
        
        with open(cfg.METADATA_DIR / f"{ticker}.json") as f: features = json.load(f)['features']
        close_idx = features.index('close')
        
        # 예측 수행
        predicted_price = run_inference(ticker, model_info, df_recent, date, cfg)
        if predicted_price is None:
            continue
        
        # 결과 저장 (로그 변환된 값 -> 원래 가격으로 변환)
        actual_price_log = df_actual.iloc[0][features].iloc[close_idx]
        last_close_log = df_recent.iloc[-1][features].iloc[close_idx]
        
        predictions.append(predicted_price)
        actuals.append(np.expm1(actual_price_log))
        last_closes.append(np.expm1(last_close_log))

    if not predictions:
        return {'backtest_f1': 0, 'backtest_nrmse': 1.0, 'plot_path': 'N/A'}
    
    # 성능 계산 (F1-Score, NRMSE)
    y_true, y_pred, y_last = np.array(actuals), np.array(predictions), np.array(last_closes)
    true_dir, pred_dir = (y_true > y_last).astype(int), (y_pred > y_last).astype(int)
    f1 = f1_score(true_dir, pred_dir, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 1e-5 else 0

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 시각화 부분 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    try:
        # --- 차트 데이터 확장 ---
        # 1. x축(날짜)을 하루 더 확장
        last_backtest_date = backtest_dates.iloc[-1]
        next_pred_date = get_next_trading_day(last_backtest_date, all_trading_dates)
        plot_dates = backtest_dates[-len(y_pred):].tolist() + [next_pred_date]
        
        # 2. y축(예측가)에 '내일 예측가' 추가
        plot_predictions = y_pred.tolist() + [next_day_prediction]

        # --- 시각화 ---
        plt.style.use('seaborn-v0_8-darkgrid')
        # ... (폰트 설정 부분은 이전 답변과 동일) ...
        plt.rc('font', family='Malgun Gothic' if sys.platform.startswith('win') else 'AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(12, 6))
        # 실제 주가는 기존대로 표시
        ax.plot(backtest_dates[-len(y_true):], y_true, label='실제 주가 (Actual Price)', marker='o', markersize=4, linestyle='-')
        # 예측 주가는 확장된 데이터로 표시
        ax.plot(plot_dates, plot_predictions, label='예측 주가 (Predicted Price)', marker='x', markersize=4, linestyle='--')
        
        ax.set_title(f'최근 {backtest_days}일 백테스트: {ticker_name} ({ticker})', fontsize=16)
        ax.set_ylabel('종가 (KRW)', fontsize=12)
        ax.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = cfg.BACKTEST_PLOTS_DIR / f"{ticker}_backtest.png"
        plt.savefig(plot_path)
        plt.close(fig)

    except Exception as e:
        print(f"Plotting Error for {ticker}: {e}")
        plot_path = "N/A"
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    return {'backtest_f1': round(f1, 4), 'backtest_nrmse': round(nrmse, 4), 'plot_path': str(plot_path)}
'''    
    

def predict_worker(args: Tuple) -> List[Dict]:
    #ticker_info, dates_to_predict, all_trading_dates = args
    #cfg = Config()
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 인자로 받은 cfg를 사용
    ticker_info, dates_to_predict, all_trading_dates, cfg = args
    # cfg = Config() # <-- 이 줄을 반드시 삭제!
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    ticker = ticker_info['ticker']
    
    results = []
    
    try:
        data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
        model_seq_len = MODEL_CONFIGS[ticker_info['config_name']].seq_len
        
        for base_date in dates_to_predict:
            base_date_str = base_date.strftime('%Y-%m-%d')
            
            df_recent = data_handler.get_price_data_until(ticker, base_date_str, model_seq_len)
            
            if df_recent is None or len(df_recent) < model_seq_len:
                results.append({'ticker': ticker, 'base_date': base_date_str, 'status': 'Failure', 'reason': 'Insufficient data'})
                continue

            last_known_date = df_recent['date'].iloc[-1]
            next_trading_day = get_next_trading_day(last_known_date, all_trading_dates)
            
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 디버깅 코드 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # 특정 종목과 날짜에 대해서만 출력하여 로그가 너무 많아지는 것을 방지
            if ticker == '459580' and '2025-08-05' in base_date_str:
                print(f"\n[DEBUG] Ticker: {ticker}")
                print(f"  - 현재 루프의 base_date: {base_date.date()}")
                print(f"  - DB에서 가져온 데이터의 마지막 날짜 (last_known_date): {last_known_date.date()}")
                print(f"  - 계산된 다음 영업일 (next_trading_day): {next_trading_day.date()}\n")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 추가 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            predicted_price = run_inference(ticker, ticker_info, df_recent, next_trading_day, cfg)
            
            if predicted_price is None:
                results.append({'ticker': ticker, 'base_date': base_date_str, 'status': 'Failure', 'reason': 'Inference failed'})
                continue
                
            with open(cfg.METADATA_DIR / f"{ticker}.json") as f: features = json.load(f)['features']
            close_idx = features.index('close')
            last_close_log = df_recent.iloc[-1][features].iloc[close_idx]
            last_close_price = np.expm1(last_close_log)
            
            results.append({
                'ticker': ticker,
                'base_date': base_date_str,
                'prediction_date': next_trading_day.strftime('%Y-%m-%d'),
                'last_close': last_close_price,
                'predicted_close': predicted_price,
                'expected_return': ((predicted_price / last_close_price) - 1) * 100 if last_close_price > 0 else 0,
                'status': 'Success'
            })
            
        return results
    except Exception as e:
        return [{'ticker': ticker, 'status': 'Failure', 'reason': f'Worker Error: {str(e)}'}]
    finally:
        if 'cuda' in str(cfg.device): gc.collect(); torch.cuda.empty_cache()

# ==================================================
# 6. 메인 실행 로직
# ==================================================
def find_best_models_across_versions(cfg: Config, stock_meta: Dict) -> List[Dict]:
    print("\n" + "="*20 + " STEP 1: v13 & v14 최고 성능 모델 통합 선정 " + "="*20)
    
    log_files = {'v13': cfg.V13_RESULT_DIR / "performance_log.csv", 'v14': cfg.V14_RESULT_DIR / "performance_log.csv"}
    all_logs = []
    for version, path in log_files.items():
        if path.exists():
            df = pd.read_csv(path)
            df['ticker'] = df['ticker'].astype(str) # <--- 이 줄을 추가해주세요.
            df['version'] = version
            all_logs.append(df)
            print(f"  ✓ {version} 로그 로드 완료 ({len(df)}개 기록)")
        else:
            print(f"  - {version} 로그 파일 없음. 건너뜁니다.")

    if not all_logs: sys.exit("❌ 분석할 성능 로그 파일이 없습니다.")
        
    combined_df = pd.concat(all_logs, ignore_index=True)
    stock_df = combined_df[combined_df['model_type'] == 'stock'].copy()
    
    stock_df_sorted = stock_df.sort_values(by=['f1', 'nrmse'], ascending=[False, True])
    best_models_df = stock_df_sorted.drop_duplicates('ticker', keep='first')
    
    targets = []
    for _, row in best_models_df.iterrows():
        version = row['version']
        models_dir = cfg.V13_RESULT_DIR / "models" if version == 'v13' else cfg.V14_RESULT_DIR / "models"
        ticker_meta = stock_meta.get(row['ticker'], {})
        
        targets.append({
            'ticker': row['ticker'],
            'name': ticker_meta.get('name', 'N/A'),
            'market': ticker_meta.get('market', 'N/A'),
            'sector': ticker_meta.get('sector', 'Unknown'),
            'config_name': row['config_name'],
            'f1': row['f1'],
            'nrmse': row['nrmse'],
            'version': version,
            'model_path': models_dir / f"model_{row['ticker']}_{row['config_name']}.pth"
        })
        
    print(f"✅ 총 {len(targets)}개 종목에 대한 최종 베스트 모델 선정을 완료했습니다.")
    return targets

def predict_market_indices(cfg: Config, all_trading_dates: pd.Series) -> Tuple[str, pd.DataFrame, float]:
    print("\n" + "="*20 + " STEP 2: 시장 지수 예측 및 방향성 판단 " + "="*20)
    index_predictions = []
    for ticker in ['KOSPI', 'KOSDAQ']:
        model_info = {'config_name': 'base', 'model_path': cfg.V13_RESULT_DIR / "models" / f"model_{ticker}_base.pth"}
        data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
        # [수정] .iloc[-1] 사용
        df_recent = data_handler.get_price_data_until(ticker, all_trading_dates.iloc[-1].strftime('%Y-%m-%d'), MODEL_CONFIGS['base'].seq_len)
        if df_recent is None:
            print(f"  - {ticker} 데이터 부족으로 예측 실패."); continue
        last_known_date = df_recent['date'].iloc[-1]
        next_trading_day = get_next_trading_day(last_known_date, all_trading_dates)
        predicted_price = run_inference(ticker, model_info, df_recent, next_trading_day, cfg)
        if predicted_price:
            with open(cfg.METADATA_DIR / f"{ticker}.json") as f: features = json.load(f)['features']
            close_idx = features.index('close')
            last_close_log = df_recent.iloc[-1][features].iloc[close_idx]
            last_close_price = np.expm1(last_close_log)
            expected_return = ((predicted_price / last_close_price) - 1) * 100
            index_predictions.append({'ticker': ticker, 'predicted_return': expected_return})
    if not index_predictions:
        print("  - 시장 지수 예측 실패. '보합(Neutral)'으로 간주합니다.")
        return 'Neutral', pd.DataFrame(), 0.0
    df_index = pd.DataFrame(index_predictions)
    avg_return = df_index['predicted_return'].mean()
    market_sentiment = 'Neutral'
    if avg_return >= cfg.MARKET_UP_THRESHOLD: market_sentiment = 'Bullish'
    elif avg_return <= cfg.MARKET_DOWN_THRESHOLD: market_sentiment = 'Bearish'
    print(f"  - 평균 예상 수익률: {avg_return:.2f}%"); print(f"  - 시장 방향성 판단: {market_sentiment}")
    return market_sentiment, df_index, avg_return

def diversify_by_sector(df: pd.DataFrame, max_per_sector: int) -> pd.DataFrame:
    if 'sector' not in df.columns or df['sector'].nunique() <= 1:
        return df
    
    diversified_list = []
    sector_count = {}
    df_copy = df.copy()
    for index, row in df_copy.iterrows():
        sector = row.get('sector', 'Unknown')
        if sector_count.get(sector, 0) < max_per_sector:
            diversified_list.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1
    return pd.DataFrame(diversified_list)
    
def get_prediction_for_date(ticker: str, model_info: Dict, target_date: datetime, all_trading_dates: pd.Series, cfg: Config) -> Optional[float]:
    """
    특정 'target_date'의 종가를 예측합니다.
    (내부적으로 target_date의 '이전 거래일'까지의 데이터를 사용합니다)
    """
    data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
    model_seq_len = MODEL_CONFIGS[model_info['config_name']].seq_len
    
    # target_date의 이전 거래일 찾기
    # all_trading_dates는 정렬되어 있다고 가정합니다.
    try:
        # np.where를 사용하여 target_date의 인덱스를 찾고 1을 빼서 이전 인덱스를 구합니다.
        prev_day_index = np.where(all_trading_dates.to_numpy() == np.datetime64(target_date))[0][0] - 1
        if prev_day_index < 0:
            return None
        prev_trading_day = all_trading_dates.iloc[prev_day_index]
    except IndexError:
        # target_date가 리스트에 없는 경우 등 예외 처리
        return None
    
    # 이전 거래일까지의 데이터를 가져옴
    df_recent = data_handler.get_price_data_until(ticker, prev_trading_day.strftime('%Y-%m-%d'), model_seq_len)
    if df_recent is None or len(df_recent) < model_seq_len:
        return None
        
    # 예측 실행 (예측 목표일은 target_date가 됩니다)
    predicted_price = run_inference(ticker, model_info, df_recent, target_date, cfg)
    return predicted_price

def main():
    start_time = datetime.now()
    print(f"🚀 FEDformer PPO 데이터 생성 및 예측 파이프라인 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    cfg = Config()
    print(f"▶️ Using device: {cfg.device}")
    
    data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
    stock_metadata = data_handler.get_stock_metadata()
    all_trading_dates = data_handler.get_all_trading_dates()

    if all_trading_dates.empty:
        sys.exit("❌ 'processed_daily_prices' 테이블에 데이터가 없습니다. 전처리 스크립트를 먼저 실행하세요.")

    print("\n" + "="*20 + " 예측 기간 설정 " + "="*20)
    # [수정] .iloc[-1] 사용
    latest_date_str = all_trading_dates.iloc[-1].strftime('%Y-%m-%d')
    start_date_str = input(f"▶️ 예측 시작일(YYYY-MM-DD) 입력 (미입력 시 최근 1일): ") or latest_date_str
    end_date_str = input(f"▶️ 예측 종료일(YYYY-MM-DD) 입력 (미입력 시 최신일): ") or latest_date_str
    
    dates_to_predict = all_trading_dates[(all_trading_dates >= start_date_str) & (all_trading_dates <= end_date_str)]
    if dates_to_predict.empty:
        sys.exit(f"❌ 해당 기간에 거래일이 없습니다.")
        
    # [수정] .iloc[0], .iloc[-1] 사용
    print(f"-> 예측 대상 기간: {dates_to_predict.iloc[0].date()} ~ {dates_to_predict.iloc[-1].date()} ({len(dates_to_predict)} 거래일)")

    target_models = find_best_models_across_versions(cfg, stock_metadata)
    
    print("\n" + "="*20 + " STEP 2: 기간 예측 병렬 실행 " + "="*20)
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 인자에 cfg를 추가
    worker_args = [(info, dates_to_predict, all_trading_dates, cfg) for info in target_models]
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    #worker_args = [(info, dates_to_predict, all_trading_dates) for info in target_models]

    all_results_list = []
    with mp.Pool(processes=cfg.MAX_CONCURRENT_PROCESSES) as pool:
        for res_list in tqdm(pool.imap_unordered(predict_worker, worker_args), total=len(worker_args), desc="   종목별 기간 예측"):
            if res_list: all_results_list.extend(res_list)
            
    print("\n✅ 모든 기간 예측 작업이 완료되었습니다.")
    if not all_results_list:
        print("ℹ️ 유효한 예측 결과가 없습니다."); return
        
    print("\n" + "="*20 + " STEP 3: PPO용 DB 저장 및 추천주 선정 " + "="*20)
    df_results = pd.DataFrame(all_results_list)
    
    try:
        with sqlite3.connect(cfg.PPO_DB_PATH) as conn:
            df_results.to_sql('predictions', conn, if_exists='append', index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON predictions (ticker, base_date)")
            conn.execute("""
                DELETE FROM predictions
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM predictions
                    GROUP BY ticker, base_date
                )
            """)
        print(f"  ✓ 총 {len(df_results)}건의 예측 결과를 PPO용 DB '{cfg.PPO_DB_PATH.name}'에 누적 저장했습니다.")
    except Exception as e:
        print(f"  ✗ PPO용 DB 저장 실패: {e}")

    # --- 최종일 기준 추천 로직 ---
    df_last_day_preds = df_results[df_results['base_date'] == end_date_str].copy()
    if df_last_day_preds.empty:
        print("  - 최종일에 대한 예측 결과가 없어 추천을 건너뜁니다.")
        return
        
    market_sentiment, df_index_pred, avg_market_return = predict_market_indices(cfg, all_trading_dates)
    if market_sentiment == 'Bearish':
        print("\n" + "!"*50); print("⚠️  경고: 시장 하락이 예상됩니다. 보수적인 투자를 권장합니다."); print("!"*50)
    
    df_success = df_last_day_preds[df_last_day_preds['status'] == 'Success'].copy()
    
    # [수정] target_models를 DataFrame으로 변환하여 join 준비
    df_meta = pd.DataFrame(target_models)
    # 중복 컬럼(name, market, sector)이 있다면 먼저 제거
    df_success = df_success.drop(columns=['name', 'market', 'sector'], errors='ignore')
    df_success = pd.merge(df_success, df_meta, on='ticker', how='left')

    df_success['avg_volume_20d'] = df_success['ticker'].apply(lambda t: DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT).get_avg_volume(t))
    df_filtered = df_success[df_success['avg_volume_20d'] > cfg.MIN_AVG_VOLUME].copy()
    
    #NRMSE와 F1 Score 기준으로 추가 필터링
    nrmse_threshold = 0.25 # NRMSE 허용 임계값 (원하는 값으로 조절)
    f1_threshold = 0.52    # F1 Score 최소 임계값 (원하는 값으로 조절)
    
    print(f"\n-> 필터링 조건: NRMSE < {nrmse_threshold}, F1 Score >= {f1_threshold}")
    df_filtered = df_filtered[(df_filtered['nrmse'] < nrmse_threshold) & (df_filtered['f1'] >= f1_threshold)]
    print(f"-> NRMSE/F1 필터링 후 남은 종목 수: {len(df_filtered)}개")
    
    # --- 예측 추세 기반 수익률 계산 (가장 중요한 로직) ---
    print("\n-> 예측 추세 기반 수익률 계산 중...")
    pred_trend_returns = []
    # 최종 예측 기준일(오늘)을 datetime 객체로 변환
    end_date_dt = pd.to_datetime(end_date_str)

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="   예측 추세 계산"):
        # Pred_T+1 (내일 예측가)는 이미 계산되어 있음
        pred_t_plus_1 = row['predicted_close']
        
        # Pred_T (오늘 예측가)를 새로 계산
        model_info = {'config_name': row['config_name'], 'model_path': row['model_path']}
        pred_t = get_prediction_for_date(row['ticker'], model_info, end_date_dt, all_trading_dates, cfg)
        
        if pred_t is not None and pred_t > 0:
            # (내일 예측가 / 오늘 예측가 - 1) * 100
            trend_return = ((pred_t_plus_1 / pred_t) - 1) * 100
            pred_trend_returns.append(trend_return)
        else:
            # 계산 불가시 0%로 처리하여 추천에서 제외
            pred_trend_returns.append(0)

    # 계산된 예측 추세 수익률을 새 컬럼으로 추가
    df_filtered['pred_trend_return'] = pred_trend_returns

    # --- 새로운 수익률 기반으로 필터링 및 정렬 ---
    print("\n-> 상승 추세 예측 종목 필터링...")
    # "상승"이 예상되는 종목만 필터링 (pred_trend_return > 0)
    df_filtered = df_filtered[df_filtered['pred_trend_return'] > 0]
    print(f"-> 상승 추세 필터링 후 남은 종목 수: {len(df_filtered)}개")

    # 예측된 상승률이 높은 순서대로 정렬
    df_filtered = df_filtered.sort_values('pred_trend_return', ascending=False)
    
    # --- 시장 상황에 따라 최종 추천 개수 결정 ---
    if market_sentiment == 'Bullish': num_recommend = cfg.RECOMMEND_COUNT_BULL
    elif market_sentiment == 'Bearish': num_recommend = cfg.RECOMMEND_COUNT_BEAR
    else: num_recommend = cfg.RECOMMEND_COUNT_NEUTRAL
        
    df_top_n = df_filtered.head(num_recommend * 2)
    df_recommended = diversify_by_sector(df_top_n, max_per_sector=cfg.MAX_PER_SECTOR).copy()
    df_recommended = df_recommended.head(num_recommend)
    
    if not df_recommended.empty:
        print("\n" + "="*20 + " STEP 4: 추천주 백테스팅 및 시각화 " + "="*20)
        
        # [핵심 추가] 백테스팅 실행 및 결과 병합
        backtest_results = []
        for _, row in tqdm(df_recommended.iterrows(), total=len(df_recommended), desc="   추천주 백테스팅"):
            model_info = {'config_name': row['config_name'], 'model_path': row['model_path']}
            
            #result = run_backtest_and_visualize(row['ticker'], row['name'], model_info, all_trading_dates, cfg, backtest_days=20)
            
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # '내일 예측가'인 row['predicted_close']를 추가로 전달
            result = run_backtest_and_visualize(
                row['ticker'], 
                row['name'], 
                model_info, 
                all_trading_dates, 
                cfg, 
                row['predicted_close'], # 6번째 인자 (next_day_prediction으로 전달됨)
                backtest_days=20      # 7번째 인자 (키워드로 전달됨)
            )
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            result['ticker'] = row['ticker']
            backtest_results.append(result)

        if backtest_results:
            df_backtest = pd.DataFrame(backtest_results)
            df_recommended = pd.merge(df_recommended, df_backtest, on='ticker', how='left')

        df_recommended.loc[:, 'confidence'] = df_recommended['f1'].apply(lambda x: '⭐⭐⭐' if x > 0.6 else '⭐⭐' if x > 0.55 else '⭐')
    
    print(f"-> 시장 상황({market_sentiment}) 및 필터링 후 최종 {len(df_recommended)}개 종목을 추천합니다.")

    # 엑셀 요약 시트에 들어갈 데이터 생성
    summary_data = {
        '항목': ['예측 기준일', '시장 방향성 예측', 'KOSPI/KOSDAQ 평균 수익률(%)', '상승장 기준(%)', '하락장 기준(%)', '최종 추천 종목 수'],
        '내용': [
            end_date_str, 
            market_sentiment, 
            f"{avg_market_return:.2f}", 
            f">= {cfg.MARKET_UP_THRESHOLD}", 
            f"<= {cfg.MARKET_DOWN_THRESHOLD}", 
            len(df_recommended)
        ]
    }
    
    try:
        with pd.ExcelWriter(cfg.RECOMMENDATION_EXCEL_PATH, engine='openpyxl') as writer:
            # [수정] 각 DataFrame이 비어있지 않을 때만 시트를 생성하도록 변경
            
            # 시트 1: 요약 정보
            #if 'summary_data' in locals() and summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Prediction_Summary', index=False)
            
            # 추천 종목 시트 저장 시 컬럼명 변경
            if not df_recommended.empty:
                rec_cols = [
                    'prediction_date', 'ticker', 'name', 'sector', 
                    'pred_trend_return',  # <--- 'expected_return'을 이것으로 변경
                    'confidence', 'f1', 'nrmse', 
                    'backtest_f1', 'backtest_nrmse',
                    'predicted_close', 'last_close', 'avg_volume_20d', 'market', 
                    'version', 'config_name', 'plot_path'
                ]
                cols_to_save = [c for c in rec_cols if c in df_recommended.columns]
                df_recommended[cols_to_save].to_excel(writer, sheet_name='Top_Recommendations', index=False)

            # 시트 3: 최종일 전체 예측 결과
            if not df_last_day_preds.empty:
                df_last_day_preds.to_excel(writer, sheet_name='Last_Day_All_Results', index=False)
            
            # 시트 4: 시장 지수 예측 결과
            if not df_index_pred.empty:
                df_index_pred.to_excel(writer, sheet_name='Market_Index_Prediction', index=False)
                
        print(f"📄 최종 추천 리포트를 '{cfg.RECOMMENDATION_EXCEL_PATH}'에 저장했습니다.")
    except Exception as e:
        print(f"❌ 엑셀 파일 저장 중 오류 발생: {e}")
        
        
    
    end_time = datetime.now()
    print(f"\n\n🏁 모든 작업 완료! (소요 시간: {end_time - start_time})")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
