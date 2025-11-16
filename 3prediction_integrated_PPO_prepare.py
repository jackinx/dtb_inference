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
        # 현재 스크립트의 프로젝트 루트
        self.PROJECT_ROOT = PROJECT_ROOT
        
        # 결과 디렉토리 (현재 프로젝트 내)
        self.V13_RESULT_DIR = self.PROJECT_ROOT / "training_results_v13"
        self.V14_RESULT_DIR = self.PROJECT_ROOT / "training_results_v14_retrain"
        self.PROCESSED_DATA_DIR = self.PROJECT_ROOT / "processed_data_v9"
        
        self.PROCESSED_DB_PATH = self.PROCESSED_DATA_DIR / "processed_stock_data.db"
        self.METADATA_DIR = self.PROCESSED_DATA_DIR / "metadata"
        self.SCALER_DIR = self.PROCESSED_DATA_DIR / "scalers"
        
        # 실행 시점 기준의 고유한 타임스탬프 생성
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # PPO 예측 결과 상위 폴더
        predictions_base_dir = self.PROJECT_ROOT / "predictions_v22_ppo"
        
        # 타임스탬프를 이름으로 하는 실행 결과 폴더 생성
        self.RUN_OUTPUT_DIR = predictions_base_dir / run_timestamp
        self.RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 각 결과물의 경로
        self.BACKTEST_PLOTS_DIR = self.RUN_OUTPUT_DIR / "backtest_plots"
        self.BACKTEST_PLOTS_DIR.mkdir(exist_ok=True)
        
        self.RECOMMENDATION_EXCEL_PATH = self.RUN_OUTPUT_DIR / "final_recommendations.xlsx"
        self.PPO_DB_PATH = predictions_base_dir / "ppo_predictions.db"
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 전체 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # SOURCE_DB_PATH: 원본 주가 데이터 (stock_data.db)
        # 현재 프로젝트 폴더 내에서 찾고, 없으면 부모 폴더에서 찾기
        self.SOURCE_DB_PATH = self.PROJECT_ROOT / "stock_data.db"
        
        if not self.SOURCE_DB_PATH.exists():
            # 현재 폴더에 없으면 부모 폴더의 다른 프로젝트들 검색
            parent_dir = self.PROJECT_ROOT.parent
            
            # 가능한 프로젝트 폴더명 리스트 (우선순위 순)
            possible_folders = [
                "DTB_project",
                "DeepTrader_baekdoosan", 
                "stock_analysis"
            ]
            
            for folder_name in possible_folders:
                candidate_path = parent_dir / folder_name / "stock_data.db"
                if candidate_path.exists():
                    self.SOURCE_DB_PATH = candidate_path
                    print(f"✓ stock_data.db 발견: {candidate_path}")
                    break
            
            # 그래도 못 찾으면 현재 폴더 기준으로 설정 (에러는 나중에 발생)
            if not self.SOURCE_DB_PATH.exists():
                print(f"⚠️  경고: stock_data.db를 찾을 수 없습니다.")
                print(f"   현재 경로로 설정: {self.SOURCE_DB_PATH}")
                print(f"   실행 전 파일 존재 여부를 확인하세요.")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        self.MAX_CONCURRENT_PROCESSES = max(1, mp.cpu_count() // 2)
        self.DB_TIMEOUT = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.RECOMMEND_COUNT_BULL = 7
        self.RECOMMEND_COUNT_NEUTRAL = 5
        self.RECOMMEND_COUNT_BEAR = 3
        self.MARKET_UP_THRESHOLD = 0.2
        self.MARKET_DOWN_THRESHOLD = -0.3
        
        self.MIN_AVG_VOLUME = 100000
        self.MAX_PER_SECTOR = 2

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
        if not model_path.exists(): 
            return None
        
        # 체크포인트 로드해서 학습된 feature 개수 확인
        checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
        saved_weight_shape = checkpoint['model_state_dict']['enc_embedding.value_embedding.tokenConv.weight'].shape
        trained_features_count = saved_weight_shape[1]  # 18
        
        with suppress_stdout():
            model_config = copy.deepcopy(MODEL_CONFIGS[config_name])
            
            # 메타데이터에서 전체 features 가져오기
            with open(cfg.METADATA_DIR / f"{ticker}.json", 'r') as f: 
                metadata = json.load(f)
                all_features = metadata['features']
            
            # 학습시와 동일한 개수의 features만 사용 (처음 18개)
            features = all_features[:trained_features_count]
            
            # 모델 설정 - 18개로 맞춤!
            model_config.enc_in = model_config.dec_in = trained_features_count  # 28이 아니라 18
            model_config.c_out = 1
            
            # FEDformer_base 모델 생성
            model = FEDformer_base(model_config).to(cfg.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

        # Scaler 로드
        scaler = joblib.load(cfg.SCALER_DIR / f"{ticker}.pkl")
        
        # scaler의 feature 개수도 확인
        if hasattr(scaler, 'n_features_in_'):
            if scaler.n_features_in_ != trained_features_count:
                # Scaler가 28개로 학습됐다면 처음 18개만 사용
                features = all_features[:trained_features_count]
        
        # 데이터 전처리 - 학습시 사용된 18개 features만 선택
        data_to_scale = df_recent[features].astype(np.float32)
        
        # Scaler가 28개 feature를 기대한다면 dummy 배열 생성
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > trained_features_count:
            
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [WARNING 수정] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # scaler가 feature name을 가지고 학습되었는지 확인
            if hasattr(scaler, 'feature_names_in_'):
                # 1. scaler가 기대하는 전체 feature name (예: 28개)
                scaler_features = scaler.feature_names_in_
                
                # 2. 0으로 채워진 (N, 28) DataFrame 생성 (NumPy 배열 대신)
                full_df = pd.DataFrame(0.0, index=data_to_scale.index, columns=scaler_features)
                
                # 3. 이 DataFrame에 18개 feature 값 복사
                #    (features 리스트의 이름이 scaler_features에 포함되어 있어야 함)
                try:
                    full_df[features] = data_to_scale
                except ValueError:
                    # 혹시 모를 불일치 시, 강제로 값만 복사 (경고가 다시 발생할 수 있음)
                    full_df.iloc[:, :trained_features_count] = data_to_scale.values
                
                # 4. NumPy 배열(full_array) 대신 DataFrame(full_df)을 transform
                scaled_data_full = scaler.transform(full_df) # <-- No Warning
                
                # 5. 변환된 NumPy 배열에서 18개 feature만 다시 슬라이싱
                scaled_data = scaled_data_full[:, :trained_features_count]
            
            else:
                # scaler가 feature name 없이 학습된 경우 (경고가 계속 나올 수 있음)
                full_array = np.zeros((len(data_to_scale), scaler.n_features_in_))
                # .values 사용 (제안하신 내용 적용)
                full_array[:, :trained_features_count] = data_to_scale.values 
                scaled_data = scaler.transform(full_array)[:, :trained_features_count]
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [WARNING 수정 끝] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                
        else:
            # scaler의 feature 개수(18)와 모델의 feature 개수(18)가 동일한 경우
            # data_to_scale (DataFrame)을 그대로 사용하면 경고가 발생하지 않습니다.
            scaled_data = scaler.transform(data_to_scale)
        
        # 시간 특징 생성
        time_marks = time_features(pd.DatetimeIndex(df_recent['date']), freq=model_config.freq).transpose()
        
        # 텐서 변환
        x_enc = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        x_mark_enc = torch.tensor(time_marks, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        
        # 디코더 입력 생성
        dec_inp_context = x_enc[:, -model_config.label_len:, :]
        dec_inp_zeros = torch.zeros(1, model_config.pred_len, model_config.dec_in, device=cfg.device)
        x_dec = torch.cat([dec_inp_context, dec_inp_zeros], dim=1)
        
        # 디코더 시간 특징
        decoder_dates = pd.DatetimeIndex(df_recent['date'].iloc[-model_config.label_len:].tolist() + [next_trading_day])
        x_mark_dec = torch.tensor(time_features(decoder_dates, freq=model_config.freq).T, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        
        # 추론 실행
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred_scaled = output[0, -1, 0].item()
        
        # 역변환
        close_idx = features.index('close')
        return inverse_transform_price(scaler, pred_scaled, close_idx)
        
    except Exception as e:
        if ticker in ['005930', '000660', '035720']:
            print(f"Error in run_inference for {ticker}: {str(e)}")
        return None
        
'''
def run_inference(ticker: str, model_info: Dict, df_recent: pd.DataFrame, next_trading_day: datetime, cfg: Config) -> Optional[float]:
    try:
        config_name = model_info['config_name']
        model_path = model_info['model_path']
        if not model_path.exists(): 
            return None
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
        
        # 학습시 사용된 feature 개수 확인
        saved_weight_shape = checkpoint['model_state_dict']['enc_embedding.value_embedding.tokenConv.weight'].shape
        trained_features_count = saved_weight_shape[1]  # 18
        
        with suppress_stdout():
            model_config = copy.deepcopy(MODEL_CONFIGS[config_name])
            
            # 메타데이터에서 전체 features 가져오기
            with open(cfg.METADATA_DIR / f"{ticker}.json", 'r') as f: 
                metadata = json.load(f)
                all_features = metadata['features']
            
            # 학습시와 동일한 개수의 features만 사용 (처음 18개)
            features = all_features[:trained_features_count]
            
            # 모델 설정
            model_config.enc_in = model_config.dec_in = trained_features_count
            model_config.c_out = 1
            
            # FEDformer_base 모델 생성 (FEDformerWithEmbedding 아님!)
            model = FEDformer_base(model_config).to(cfg.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

        # Scaler 로드 및 확인
        scaler = joblib.load(cfg.SCALER_DIR / f"{ticker}.pkl")
        
        # Scaler가 기대하는 feature 개수 확인
        if hasattr(scaler, 'n_features_in_'):
            if scaler.n_features_in_ != trained_features_count:
                # Scaler의 feature 개수가 다른 경우
                features = all_features[:scaler.n_features_in_]
        
        # 데이터 전처리 - 학습시 사용된 features만 선택
        data_to_scale = df_recent[features].astype(np.float32)
        scaled_data = scaler.transform(data_to_scale)
        
        # 시간 특징 생성
        time_marks = time_features(pd.DatetimeIndex(df_recent['date']), freq=model_config.freq).transpose()
        
        # 텐서 변환
        x_enc = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        x_mark_enc = torch.tensor(time_marks, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        
        # 디코더 입력 생성
        dec_inp_context = x_enc[:, -model_config.label_len:, :]
        dec_inp_zeros = torch.zeros(1, model_config.pred_len, model_config.dec_in, device=cfg.device)
        x_dec = torch.cat([dec_inp_context, dec_inp_zeros], dim=1)
        
        # 디코더 시간 특징
        decoder_dates = pd.DatetimeIndex(df_recent['date'].iloc[-model_config.label_len:].tolist() + [next_trading_day])
        x_mark_dec = torch.tensor(time_features(decoder_dates, freq=model_config.freq).T, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        
        # 추론 실행
        with torch.no_grad():
            # FEDformer_base는 4개 인자만 받음
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred_scaled = output[0, -1, 0].item()
        
        # 역변환
        close_idx = features.index('close')
        return inverse_transform_price(scaler, pred_scaled, close_idx)
        
    except Exception as e:
        # 디버깅용 - 처음 몇 개 종목만 에러 출력
        if ticker in ['005930', '000660', '035720']:
            print(f"Error in run_inference for {ticker}: {str(e)}")
        return None       
'''


def test_inference():
    """run_inference 함수 테스트"""
    print("\n" + "="*50)
    print("run_inference 함수 테스트 시작")
    print("="*50)
    
    cfg = Config()
    
    # 테스트할 종목 선택
    test_ticker = '005930'  # 삼성전자
    
    # 모델 정보 설정
    model_info = {
        'config_name': 'base',
        'model_path': cfg.V13_RESULT_DIR / "models" / f"model_{test_ticker}_base.pth"
    }
    
    print(f"테스트 종목: {test_ticker}")
    print(f"모델 경로: {model_info['model_path']}")
    print(f"모델 존재: {model_info['model_path'].exists()}")
    
    if not model_info['model_path'].exists():
        print("❌ 모델 파일이 없습니다!")
        return
    
    # 데이터 가져오기
    data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
    all_trading_dates = data_handler.get_all_trading_dates()
    
    # 최근 60일 데이터 가져오기
    df_recent = data_handler.get_price_data_until(
        test_ticker, 
        all_trading_dates.iloc[-1].strftime('%Y-%m-%d'), 
        60
    )
    
    if df_recent is None:
        print("❌ 데이터를 가져올 수 없습니다!")
        return
        
    print(f"데이터 shape: {df_recent.shape}")
    
    # 다음 거래일 계산
    next_trading_day = get_next_trading_day(df_recent['date'].iloc[-1], all_trading_dates)
    print(f"다음 거래일: {next_trading_day}")
    
    # 예측 실행 - run_inference 함수를 직접 호출
    print("\n예측 실행 중...")
    predicted_price = run_inference(test_ticker, model_info, df_recent, next_trading_day, cfg)
    
    if predicted_price is not None:
        print(f"✓ 최종 예측가격: {predicted_price:.2f}")
    else:
        print("❌ 예측 실패")
        

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
    

def predict_worker(args: Tuple) -> List[Dict]:
    ticker_info, dates_to_predict, all_trading_dates, cfg = args
    ticker = ticker_info['ticker']
    
    # 디버깅: 처음 몇 개 종목만 출력
    if ticker in ['005930', '000660', '035720']:  # 삼성전자, SK하이닉스, 카카오 등
        print(f"\n[DEBUG predict_worker] Processing {ticker}")
        print(f"  - dates_to_predict 수: {len(dates_to_predict)}")
        if len(dates_to_predict) > 0:
            print(f"  - 첫 날짜: {dates_to_predict.iloc[0]}")
            print(f"  - 마지막 날짜: {dates_to_predict.iloc[-1]}")
    
    results = []
    
    try:
        data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
        model_seq_len = MODEL_CONFIGS[ticker_info['config_name']].seq_len
        
        for base_date in dates_to_predict:
            base_date_str = base_date.strftime('%Y-%m-%d')
            
            df_recent = data_handler.get_price_data_until(ticker, base_date_str, model_seq_len)
            
            if df_recent is None or len(df_recent) < model_seq_len:
                results.append({'ticker': ticker, 'base_date': base_date_str, 'status': 'Failure'})
                continue

            last_known_date = df_recent['date'].iloc[-1]
            next_trading_day = get_next_trading_day(last_known_date, all_trading_dates)
            
            predicted_price = run_inference(ticker, ticker_info, df_recent, next_trading_day, cfg)
            
            if predicted_price is None:
                results.append({'ticker': ticker, 'base_date': base_date_str, 'status': 'Failure'})
                continue
                
            with open(cfg.METADATA_DIR / f"{ticker}.json") as f: 
                features = json.load(f)['features']
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
                'status': 'Success',
                # 여기에 f1과 nrmse 추가!
                'f1': ticker_info.get('f1', 0),
                'nrmse': ticker_info.get('nrmse', 1.0),
                'config_name': ticker_info['config_name'],
                'model_path': str(ticker_info['model_path']),
                'name': ticker_info.get('name', 'N/A'),
                'market': ticker_info.get('market', 'N/A'),
                'sector': ticker_info.get('sector', 'Unknown')
            })
            
        return results
    except Exception as e:
        return [{'ticker': ticker, 'status': 'Failure'}]
    finally:
        if 'cuda' in str(cfg.device): 
            gc.collect()
            torch.cuda.empty_cache()

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
        model_info = {
            'config_name': 'base', 
            'model_path': cfg.V13_RESULT_DIR / "models" / f"model_{ticker}_base.pth"
        }
        
        # 모델 파일 존재 확인
        if not model_info['model_path'].exists():
            print(f"  - {ticker} 모델 파일 없음. 건너뜁니다.")
            continue
            
        data_handler = DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT)
        df_recent = data_handler.get_price_data_until(
            ticker, 
            all_trading_dates.iloc[-1].strftime('%Y-%m-%d'), 
            MODEL_CONFIGS['base'].seq_len
        )
        
        if df_recent is None:
            print(f"  - {ticker} 데이터 부족으로 예측 실패.")
            continue
            
        last_known_date = df_recent['date'].iloc[-1]
        next_trading_day = get_next_trading_day(last_known_date, all_trading_dates)
        
        # run_inference 사용
        predicted_price = run_inference(ticker, model_info, df_recent, next_trading_day, cfg)
        
        if predicted_price:
            # 체크포인트 로드해서 학습시 feature 개수 확인
            checkpoint = torch.load(model_info['model_path'], map_location=cfg.device, weights_only=False)
            trained_features_count = checkpoint['model_state_dict']['enc_embedding.value_embedding.tokenConv.weight'].shape[1]
            
            with open(cfg.METADATA_DIR / f"{ticker}.json") as f: 
                all_features = json.load(f)['features']
            
            features = all_features[:trained_features_count]
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
    if avg_return >= cfg.MARKET_UP_THRESHOLD: 
        market_sentiment = 'Bullish'
    elif avg_return <= cfg.MARKET_DOWN_THRESHOLD: 
        market_sentiment = 'Bearish'
        
    print(f"  - 평균 예상 수익률: {avg_return:.2f}%")
    print(f"  - 시장 방향성 판단: {market_sentiment}")
    
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

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 전체 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    print("\n" + "="*20 + " 예측 기간 설정 " + "="*20)
    
    # DB의 실제 최신 날짜 확인
    latest_date = all_trading_dates.iloc[-1]
    earliest_date = all_trading_dates.iloc[0]
    
    print(f"DB 데이터 범위: {earliest_date.date()} ~ {latest_date.date()}")
    latest_date_str = latest_date.strftime('%Y-%m-%d')
    
    # 사용자 입력
    start_date_input = input(f"▶️ 예측 시작일(YYYY-MM-DD) 입력 (미입력 시 최신일 {latest_date_str}): ").strip()
    end_date_input = input(f"▶️ 예측 종료일(YYYY-MM-DD) 입력 (미입력 시 최신일 {latest_date_str}): ").strip()
    
    # 기본값 설정
    start_date_str = start_date_input if start_date_input else latest_date_str
    end_date_str = end_date_input if end_date_input else latest_date_str
    
    # 문자열을 datetime으로 변환
    try:
        start_date_dt = pd.to_datetime(start_date_str)
        end_date_dt = pd.to_datetime(end_date_str)
    except ValueError as e:
        sys.exit(f"❌ 날짜 형식 오류: {e}")
    
    # ========== 핵심 수정: 거래일이 아닌 날짜 처리 ==========
    # 시작일이 거래일이 아닌 경우, 그 이전의 가장 가까운 거래일로 조정
    if start_date_dt not in all_trading_dates.values:
        prev_trading_dates = all_trading_dates[all_trading_dates <= start_date_dt]
        if prev_trading_dates.empty:
            print(f"❌ {start_date_str} 이전에 거래일이 없습니다.")
            sys.exit(1)
        adjusted_start = prev_trading_dates.iloc[-1]
        print(f"⚠️  {start_date_dt.date()}는 거래일이 아닙니다.")
        print(f"   → 이전 거래일인 {adjusted_start.date()}로 자동 조정")
        start_date_dt = adjusted_start
        start_date_str = adjusted_start.strftime('%Y-%m-%d')

    # 종료일이 거래일이 아닌 경우, 그 이전의 가장 가까운 거래일로 조정
    if end_date_dt not in all_trading_dates.values:
        prev_trading_dates = all_trading_dates[all_trading_dates <= end_date_dt]
        if prev_trading_dates.empty:
            print(f"❌ {end_date_str} 이전에 거래일이 없습니다.")
            sys.exit(1)
        adjusted_end = prev_trading_dates.iloc[-1]
        print(f"⚠️  {end_date_dt.date()}는 거래일이 아닙니다.")
        print(f"   → 이전 거래일인 {adjusted_end.date()}로 자동 조정")
        end_date_dt = adjusted_end
        end_date_str = adjusted_end.strftime('%Y-%m-%d')
        
        
    # 입력된 날짜가 DB 범위를 벗어나는지 확인
    if start_date_dt > latest_date:
        print(f"⚠️  경고: 시작일({start_date_dt.date()})이 DB 최신일({latest_date.date()})보다 미래입니다.")
        print(f"   -> 최신일로 자동 조정합니다.")
        start_date_dt = latest_date
        start_date_str = latest_date_str
    
    if end_date_dt > latest_date:
        print(f"⚠️  경고: 종료일({end_date_dt.date()})이 DB 최신일({latest_date.date()})보다 미래입니다.")
        print(f"   -> 최신일로 자동 조정합니다.")
        end_date_dt = latest_date
        end_date_str = latest_date_str
    
    # datetime 타입으로 필터링 (문자열 비교 X)
    dates_to_predict = all_trading_dates[
        (all_trading_dates >= start_date_dt) & 
        (all_trading_dates <= end_date_dt)
    ]
    
    if dates_to_predict.empty:
        print(f"❌ {start_date_str} ~ {end_date_str} 기간에 거래일이 없습니다.")
        print(f"   DB 데이터 범위를 확인하세요: {earliest_date.date()} ~ {latest_date.date()}")
        sys.exit(1)
    
    print(f"✅ 예측 대상 기간: {dates_to_predict.iloc[0].date()} ~ {dates_to_predict.iloc[-1].date()} ({len(dates_to_predict)} 거래일)")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    target_models = find_best_models_across_versions(cfg, stock_metadata)

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
            # reason 컬럼이 있으면 제거
        if 'reason' in df_results.columns:
            df_results = df_results.drop(columns=['reason'])
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

    # 디버깅 출력 추가
    print(f"\n[DEBUG] 전체 예측 결과 수: {len(df_results)}")
    print(f"[DEBUG] end_date_str: {end_date_str}")
    print(f"[DEBUG] df_results의 고유 base_date: {df_results['base_date'].unique()[:5]}...")  # 처음 5개만
    print(f"[DEBUG] 최종일({end_date_str}) 예측 결과 수: {len(df_last_day_preds)}")

    if not df_last_day_preds.empty:
        print(f"[DEBUG] df_last_day_preds columns: {df_last_day_preds.columns.tolist()}")
        print(f"[DEBUG] status 분포: {df_last_day_preds['status'].value_counts().to_dict()}")
        
        # 실패한 경우 이유 확인
        if 'reason' in df_last_day_preds.columns:
            print(f"[DEBUG] 실패 이유 분포: {df_last_day_preds[df_last_day_preds['status']=='Failure']['reason'].value_counts().head()}")

    if df_last_day_preds.empty:
        print("  - 최종일에 대한 예측 결과가 없어 추천을 건너뜁니다.")
        return
        
    market_sentiment, df_index_pred, avg_market_return = predict_market_indices(cfg, all_trading_dates)
    if market_sentiment == 'Bearish':
        print("\n" + "!"*50); print("⚠️  경고: 시장 하락이 예상됩니다. 보수적인 투자를 권장합니다."); print("!"*50)
    
    df_success = df_last_day_preds[df_last_day_preds['status'] == 'Success'].copy()

    # df_success에 이미 필요한 컬럼들이 있는지 확인
    print(f"\n[DEBUG] df_success columns: {df_success.columns.tolist()}")
    print(f"[DEBUG] df_success shape: {df_success.shape}")
    if not df_success.empty:
        print(f"[DEBUG] Sample f1 values: {df_success['f1'].head() if 'f1' in df_success.columns else 'f1 column missing'}")
        print(f"[DEBUG] Sample nrmse values: {df_success['nrmse'].head() if 'nrmse' in df_success.columns else 'nrmse column missing'}")
    

    df_success['avg_volume_20d'] = df_success['ticker'].apply(lambda t: DataHandler(cfg.PROCESSED_DB_PATH, cfg.SOURCE_DB_PATH, cfg.DB_TIMEOUT).get_avg_volume(t))
    df_filtered = df_success[df_success['avg_volume_20d'] > cfg.MIN_AVG_VOLUME].copy()
    
    # NRMSE와 F1 Score 기준으로 추가 필터링
    nrmse_threshold = 0.25
    f1_threshold = 0.52

    print(f"\n-> [1차 필터링] NRMSE < {nrmse_threshold}, F1 Score >= {f1_threshold}")
    df_filtered = df_success[(df_success['nrmse'] < nrmse_threshold) & (df_success['f1'] >= f1_threshold)]
    print(f"-> 1차 필터링 후: {len(df_filtered)}개 종목")

    # 최소 추천 개수 설정
    MIN_RECOMMENDATIONS = 3

    # 필터링 결과가 너무 적으면 조건 완화
    if len(df_filtered) < MIN_RECOMMENDATIONS:
        print(f"\n⚠️  1차 필터링 통과 종목이 {len(df_filtered)}개로 부족합니다.")
        print(f"   -> 조건을 완화하여 최소 {MIN_RECOMMENDATIONS}개 추천")
        
        # 2차 완화: NRMSE < 0.30, F1 >= 0.50
        nrmse_threshold_relaxed = 0.30
        f1_threshold_relaxed = 0.50
        
        print(f"   [2차 필터링] NRMSE < {nrmse_threshold_relaxed}, F1 >= {f1_threshold_relaxed}")
        df_filtered = df_success[
            (df_success['nrmse'] < nrmse_threshold_relaxed) & 
            (df_success['f1'] >= f1_threshold_relaxed)
        ]
        print(f"   -> 2차 필터링 후: {len(df_filtered)}개 종목")
        
        # 여전히 부족하면 F1 Score 상위 N개 선택
        if len(df_filtered) < MIN_RECOMMENDATIONS:
            print(f"\n⚠️  2차 필터링도 부족 ({len(df_filtered)}개)")
            print(f"   -> F1 Score 상위 {MIN_RECOMMENDATIONS}개를 강제 선택 (⚠️ 성능 주의)")
            
            df_filtered = df_success.sort_values('f1', ascending=False).head(MIN_RECOMMENDATIONS * 2)
            print(f"   -> 최종: {len(df_filtered)}개 종목 (성능 경고 포함)")
    

    # --- 예측 추세 기반 수익률 계산 ---
    print("\n-> 예측 추세 기반 수익률 계산 중...")
    pred_trend_returns = []
    end_date_dt = pd.to_datetime(end_date_str)

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="    예측 추세 계산"):
        pred_t_plus_1 = row['predicted_close']
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [이 부분 수정] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        model_info = {
            'config_name': row['config_name'], 
            'model_path': Path(row['model_path']) # 👈 str을 Path 객체로 다시 변환
        }
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [여기까지 수정] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        pred_t = get_prediction_for_date(row['ticker'], model_info, end_date_dt, all_trading_dates, cfg)
        
        if pred_t is not None and pred_t > 0:
            trend_return = ((pred_t_plus_1 / pred_t) - 1) * 100
            pred_trend_returns.append(trend_return)
        else:
            pred_trend_returns.append(0)

    df_filtered = df_filtered.copy()  # 복사본 생성
    df_filtered['pred_trend_return'] = pred_trend_returns
    

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 전체 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # --- 1단계: 상승 추세 필터링 (기본 조건) ---
    print("\n-> 1단계: 상승 추세 예측 종목 필터링...")
    df_uptrend = df_filtered[df_filtered['pred_trend_return'] > 0].copy()
    print(f"   ✓ 상승 추세 필터링 후: {len(df_uptrend)}개 종목")
    
    if df_uptrend.empty:
        print("   ⚠️  상승이 예상되는 종목이 없어 추천을 종료합니다.")
        return
    
    # --- 2단계: Confidence(F1 Score) 기준 정렬 ---
    print("\n-> 2단계: 신뢰도(F1 Score) 기준으로 정렬...")
    # F1 Score가 높은 순서대로 정렬 (동점일 경우 pred_trend_return으로 2차 정렬)
    df_sorted = df_uptrend.sort_values(
        by=['f1', 'pred_trend_return'], 
        ascending=[False, False]
    )
    
    # Confidence 등급 부여 (정렬 전에 미리 계산)
    df_sorted['confidence'] = df_sorted['f1'].apply(
        lambda x: '⭐⭐⭐' if x >= 0.60 else '⭐⭐' if x >= 0.55 else '⭐'
    )
    
    # 상위 confidence 종목들만 먼저 선별
    print(f"   - ⭐⭐⭐(F1≥0.60): {len(df_sorted[df_sorted['f1'] >= 0.60])}개")
    print(f"   - ⭐⭐  (F1≥0.55): {len(df_sorted[df_sorted['f1'] >= 0.55])}개")
    print(f"   - ⭐   (F1<0.55): {len(df_sorted[df_sorted['f1'] < 0.55])}개")
    
    # --- 3단계: 시장 상황에 따라 추천 개수 결정 ---
    if market_sentiment == 'Bullish': 
        num_recommend = cfg.RECOMMEND_COUNT_BULL
    elif market_sentiment == 'Bearish': 
        num_recommend = cfg.RECOMMEND_COUNT_BEAR
    else: 
        num_recommend = cfg.RECOMMEND_COUNT_NEUTRAL
    
    print(f"\n-> 3단계: 시장 상황({market_sentiment})에 따른 추천 개수: {num_recommend}개")
    
    # --- 4단계: 섹터 분산을 고려한 최종 선정 ---
    # 정렬된 상위 종목 중에서 섹터 분산 적용 (여유있게 2배 선택)
    df_top_candidates = df_sorted.head(num_recommend * 3)  # 3배로 늘려서 선택 폭 확대
    
    print(f"\n-> 4단계: 섹터 분산 적용 (후보: {len(df_top_candidates)}개)")
    df_recommended = diversify_by_sector(
        df_top_candidates, 
        max_per_sector=cfg.MAX_PER_SECTOR
    ).copy()
    
    # 최종 개수만큼만 선택
    df_recommended = df_recommended.head(num_recommend)
    
    # 섹터별 분포 출력
    if not df_recommended.empty:
        sector_dist = df_recommended['sector'].value_counts()
        print(f"   ✓ 최종 추천 종목의 섹터 분포:")
        for sector, count in sector_dist.items():
            print(f"      - {sector}: {count}개")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    print(f"\n✅ 시장 상황({market_sentiment}) 및 신뢰도 기준으로 최종 {len(df_recommended)}개 종목을 추천합니다.")
    
    if not df_recommended.empty:
        print("\n" + "="*20 + " STEP 4: 추천주 백테스팅 및 시각화 " + "="*20)
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 백테스팅 먼저 실행 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        backtest_results = []
        for _, row in tqdm(df_recommended.iterrows(), total=len(df_recommended), desc="    추천주 백테스팅"):
            
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [이 부분 수정] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            model_info = {
                'config_name': row['config_name'], 
                'model_path': Path(row['model_path']) # 👈 str을 Path 객체로 다시 변환
            }
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [여기까지 수정] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            result = run_backtest_and_visualize(
                row['ticker'], 
                row['name'], 
                model_info, 
                all_trading_dates, 
                cfg, 
                row['predicted_close'],
                backtest_days=20
            )
            result['ticker'] = row['ticker']
            backtest_results.append(result)

        if backtest_results:
            df_backtest = pd.DataFrame(backtest_results)
            df_recommended = pd.merge(df_recommended, df_backtest, on='ticker', how='left')
        
        # Confidence 등급 부여
        df_recommended['confidence'] = df_recommended['f1'].apply(
            lambda x: '⭐⭐⭐' if x >= 0.60 else '⭐⭐' if x >= 0.55 else '⭐'
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지 백테스팅 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 전체 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        print("\n" + "="*20 + " STEP 5: 백테스트 성능 기반 최종 선정 " + "="*20)
        
        # --- 1단계: 백테스트 F1과 NRMSE 기준으로 재필터링 ---
        print("\n-> 1단계: 백테스트 성능 필터링...")
        
        # STEP 5의 필터링 기준 부분 수정

        print("\n" + "="*20 + " STEP 5: 백테스트 성능 기반 최종 선정 " + "="*20)
        
        # 백테스트 성능 기준 설정
        backtest_f1_threshold = 0.55     
        backtest_nrmse_threshold = 0.15  
        relaxed_f1_threshold = 0.50      
        relaxed_nrmse_threshold = 0.20   
        
        print(f"\n-> 1단계: 백테스트 성능 필터링...")
        print(f"   [엄격 기준] backtest_f1 >= {backtest_f1_threshold}, backtest_nrmse <= {backtest_nrmse_threshold}")
        
        df_backtest_filtered = df_recommended[
            (df_recommended['backtest_f1'] >= backtest_f1_threshold) & 
            (df_recommended['backtest_nrmse'] <= backtest_nrmse_threshold)
        ].copy()
        
        print(f"   ✓ 엄격 기준 통과: {len(df_backtest_filtered)}개 종목")
        
        # 목표 개수
        if market_sentiment == 'Bullish': 
            num_final = cfg.RECOMMEND_COUNT_BULL
        elif market_sentiment == 'Bearish': 
            num_final = cfg.RECOMMEND_COUNT_BEAR
        else: 
            num_final = cfg.RECOMMEND_COUNT_NEUTRAL
        
        # 1단계: 엄격 기준으로 부족하면 추가 백테스팅
        if len(df_backtest_filtered) < num_final:
            print(f"\n   📊 엄격 기준 통과 종목이 부족합니다. 추가 백테스팅을 진행합니다.")
            
            already_tested = set(df_recommended['ticker'].tolist())
            remaining_candidates = df_sorted[~df_sorted['ticker'].isin(already_tested)].head(num_final * 3)
            
            if not remaining_candidates.empty:
                print(f"      -> 추가 {len(remaining_candidates)}개 종목 백테스팅 중...")
                
                additional_backtest_results = []
                for _, row in tqdm(remaining_candidates.iterrows(), 
                   total=len(remaining_candidates), 
                   desc="    추가 백테스팅"):
                    
                    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [이 부분 수정] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                    model_info = {
                        'config_name': row['config_name'], 
                        'model_path': Path(row['model_path']) # 👈 str을 Path 객체로 다시 변환
                    }
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [여기까지 수정] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                    
                    result = run_backtest_and_visualize(
                        row['ticker'], 
                        row['name'], 
                        model_info, 
                        all_trading_dates, 
                        cfg, 
                        row['predicted_close'],
                        backtest_days=20
                    )
                    result['ticker'] = row['ticker']
                    additional_backtest_results.append(result)
                
                if additional_backtest_results:
                    df_additional_backtest = pd.DataFrame(additional_backtest_results)
                    remaining_candidates = pd.merge(
                        remaining_candidates, 
                        df_additional_backtest, 
                        on='ticker', 
                        how='left'
                    )
                    
                    df_additional_strict = remaining_candidates[
                        (remaining_candidates['backtest_f1'] >= backtest_f1_threshold) & 
                        (remaining_candidates['backtest_nrmse'] <= backtest_nrmse_threshold)
                    ].copy()
                    
                    if not df_additional_strict.empty:
                        print(f"      ✓ 추가 {len(df_additional_strict)}개 종목이 엄격 기준 통과")
                        df_backtest_filtered = pd.concat([df_backtest_filtered, df_additional_strict], ignore_index=True)
        
        # 2단계: 여전히 부족하면 완화된 기준 적용
        if len(df_backtest_filtered) < num_final:
            print(f"\n   📊 엄격 기준 통과 종목({len(df_backtest_filtered)}개)이 여전히 부족합니다.")
            print(f"      -> 완화 기준 적용: F1 >= {relaxed_f1_threshold}, NRMSE <= {relaxed_nrmse_threshold}")
            
            all_backtested = df_recommended.copy()
            if 'remaining_candidates' in locals() and not remaining_candidates.empty:
                all_backtested = pd.concat([
                    df_recommended,
                    remaining_candidates
                ], ignore_index=True).drop_duplicates(subset=['ticker'])
            
            df_relaxed = all_backtested[
                (all_backtested['backtest_f1'] >= relaxed_f1_threshold) & 
                (all_backtested['backtest_nrmse'] <= relaxed_nrmse_threshold)
            ].copy()
            
            df_relaxed_sorted = df_relaxed.sort_values(
                by=['backtest_f1', 'backtest_nrmse'], 
                ascending=[False, True]
            )
            
            already_selected = set(df_backtest_filtered['ticker'].tolist())
            df_relaxed_additional = df_relaxed_sorted[
                ~df_relaxed_sorted['ticker'].isin(already_selected)
            ].head(num_final - len(df_backtest_filtered))
            
            if not df_relaxed_additional.empty:
                df_backtest_filtered = pd.concat([
                    df_backtest_filtered, 
                    df_relaxed_additional
                ], ignore_index=True)
                print(f"      ✓ 완화 기준으로 {len(df_relaxed_additional)}개 종목 추가 (총 {len(df_backtest_filtered)}개)")
        
        # 3단계: 그래도 부족하면 성능 순으로 선택
        if len(df_backtest_filtered) < num_final:
            print(f"\n   ⚠️  기준을 만족하는 종목이 {len(df_backtest_filtered)}개뿐입니다.")
            print(f"      -> 백테스트 성능 상위 {num_final}개를 선정합니다.")
            
            all_backtested = df_recommended.copy()
            if 'remaining_candidates' in locals() and not remaining_candidates.empty:
                all_backtested = pd.concat([
                    df_recommended,
                    remaining_candidates
                ], ignore_index=True).drop_duplicates(subset=['ticker'])
            
            df_backtest_filtered = all_backtested.sort_values(
                by=['backtest_f1', 'backtest_nrmse'], 
                ascending=[False, True]
            ).head(num_final * 2)
        
        # --- 2단계: 백테스트 F1 기준으로 정렬 ---
        print("\n-> 2단계: 백테스트 신뢰도(F1) 기준으로 정렬...")
        
        df_sorted_by_backtest = df_backtest_filtered.sort_values(
            by=['backtest_f1', 'backtest_nrmse'], 
            ascending=[False, True]
        )
        
        print(f"   - 백테스트 성능 상위 종목:")
        for idx, (_, row) in enumerate(df_sorted_by_backtest.head(min(10, len(df_sorted_by_backtest))).iterrows(), 1):
            print(f"      {idx}. {row['name']}({row['ticker']}): F1={row['backtest_f1']:.3f}, NRMSE={row['backtest_nrmse']:.3f}")
        
        # --- 3단계: 섹터 분산 후 최종 개수 선정 ---
        print(f"\n-> 3단계: 최종 {num_final}개 종목 선정...")
        
        df_top_by_backtest = df_sorted_by_backtest.head(num_final * 2)
        
        print(f"   - 섹터 분산 적용 (후보: {len(df_top_by_backtest)}개)")
        df_diversified = diversify_by_sector(
            df_top_by_backtest, 
            max_per_sector=cfg.MAX_PER_SECTOR
        )
        
        df_selected = df_diversified.head(num_final)
        
        # --- 4단계: 예상 수익률 기준으로 재정렬 ---
        print(f"\n-> 4단계: 예상 수익률 기준으로 최종 정렬...")
        
        df_final = df_selected.sort_values('pred_trend_return', ascending=False)
        
        print(f"   ✓ 최종 {len(df_final)}개 종목 선정 완료")
        print(f"\n   【최종 추천 종목 (수익률 순)】")
        for idx, (_, row) in enumerate(df_final.iterrows(), 1):
            print(f"      {idx}. {row['name']:15s} ({row['ticker']:6s}) | "
                  f"예상수익률: {row['pred_trend_return']:+6.2f}% | "
                  f"BT_F1: {row['backtest_f1']:.3f} | "
                  f"BT_NRMSE: {row['backtest_nrmse']:.3f}")
        
        if not df_final.empty:
            sector_dist = df_final['sector'].value_counts()
            print(f"\n   ✓ 최종 추천 종목의 섹터 분포:")
            for sector, count in sector_dist.items():
                print(f"      - {sector}: {count}개")
        
        df_recommended = df_final.copy()        

    # ========== Excel 파일 출력 코드 추가 ==========
    if not df_recommended.empty:
        print("\n" + "="*20 + " STEP 6: Excel 파일 저장 " + "="*20)
        
        try:
            # 다음 거래일 계산
            next_trading_day = get_next_trading_day(pd.to_datetime(end_date_str), all_trading_dates)
            
            # Excel 출력용 데이터프레임 준비
            excel_df = df_recommended[['ticker', 'name', 'market', 'sector', 
                                 'last_close', 'predicted_close', 
                                 'expected_return', 'pred_trend_return',
                                 'f1', 'nrmse', 'backtest_f1', 'backtest_nrmse', 
                                 'confidence', 'avg_volume_20d']].copy()
            
            # 예측 날짜 정보 추가
            excel_df.insert(0, '예측대상일', next_trading_day.strftime('%Y-%m-%d'))
            excel_df.insert(0, '기준일', end_date_str)
            
            # 컬럼명을 한글로 변경
            excel_df.columns = ['기준일', '예측대상일', '종목코드', '종목명', '시장', '섹터', 
                                '현재가', '예측가', 
                                '예상수익률(%)', '추세기반수익률(%)',
                                '모델F1', '모델NRMSE', '백테스트F1', '백테스트NRMSE', 
                                '신뢰도', '20일평균거래량']
            
            # 숫자 포맷팅
            excel_df['현재가'] = excel_df['현재가'].round(0).astype(int)
            excel_df['예측가'] = excel_df['예측가'].round(0).astype(int)
            excel_df['예상수익률(%)'] = excel_df['예상수익률(%)'].round(2)
            excel_df['추세기반수익률(%)'] = excel_df['추세기반수익률(%)'].round(2)
            excel_df['모델F1'] = excel_df['모델F1'].round(4)
            excel_df['모델NRMSE'] = excel_df['모델NRMSE'].round(4)
            excel_df['백테스트F1'] = excel_df['백테스트F1'].round(4)
            excel_df['백테스트NRMSE'] = excel_df['백테스트NRMSE'].round(4)
            excel_df['20일평균거래량'] = excel_df['20일평균거래량'].round(0).astype(int)
            
            # 시장 상황 정보 추가
            summary_data = {
                '항목': [
                    '기준일',
                    '예측대상일',
                    '시장상황', 
                    'KOSPI예상수익률', 
                    'KOSDAQ예상수익률', 
                    '추천종목수'
                ],
                '값': [
                    end_date_str,
                    next_trading_day.strftime('%Y-%m-%d'),
                    market_sentiment, 
                    f"{df_index_pred[df_index_pred['ticker']=='KOSPI']['predicted_return'].values[0]:.2f}%" if not df_index_pred.empty and 'KOSPI' in df_index_pred['ticker'].values else 'N/A',
                    f"{df_index_pred[df_index_pred['ticker']=='KOSDAQ']['predicted_return'].values[0]:.2f}%" if not df_index_pred.empty and 'KOSDAQ' in df_index_pred['ticker'].values else 'N/A',
                    len(df_recommended)  # df_final → df_recommended
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Excel 파일로 저장 (여러 시트)
            with pd.ExcelWriter(cfg.RECOMMENDATION_EXCEL_PATH, engine='openpyxl') as writer:
                # 요약 정보 시트
                summary_df.to_excel(writer, sheet_name='요약', index=False)
                
                # 추천 종목 시트
                excel_df.to_excel(writer, sheet_name='추천종목', index=False)
                
                # 시장 지수 예측 시트
                if not df_index_pred.empty:
                    df_index_pred.to_excel(writer, sheet_name='시장지수예측', index=False)
            
            print(f"  ✓ Excel 파일 저장 완료: {cfg.RECOMMENDATION_EXCEL_PATH}")
            print(f"    - 파일 위치: {cfg.RECOMMENDATION_EXCEL_PATH.absolute()}")
            print(f"    - 기준일: {end_date_str} → 예측대상일: {next_trading_day.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"  ✗ Excel 파일 저장 실패: {e}")
            print(f"    오류 상세: {type(e).__name__}: {str(e)}")

# ========== Excel 파일 출력 코드 끝 ==========
    end_time = datetime.now()
    print(f"\n\n🏁 모든 작업 완료! (소요 시간: {end_time - start_time})")

if __name__ == '__main__':
#    test_inference()
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
