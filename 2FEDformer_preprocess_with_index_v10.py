"""
FEDformer 데이터 전처리 스크립트 (v9 - 수정본)

주요 수정사항:
1. 모든 기술적 지표를 DB에 저장하도록 수정
2. pandas_ta 의존성 제거 (ta-lib 또는 ta 라이브러리 사용)
3. 컬럼명 정규화 (소수점 제거)
"""

import os
import sqlite3
import pandas as pd
import pandas_ta as ta

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import gc
import joblib
import json
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import sys

# ta 라이브러리 import
try:
    import ta
    USE_TA = True
except ImportError:
    print("⚠️ ta 라이브러리가 없습니다. pip install ta 실행 필요")
    USE_TA = False

# ==================================================
# 1. 설정
# ==================================================
                       
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

                               
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data_v9"
DESTINATION_DB_PATH = PROCESSED_DATA_DIR / "processed_stock_data.db"
SCALER_DIR = PROCESSED_DATA_DIR / "scalers"
METADATA_DIR = PROCESSED_DATA_DIR / "metadata"
LOG_DIR = PROCESSED_DATA_DIR / "logs"

                                             
                                                                   
                                    
                                                                                                            
SOURCE_DB_PATH = PROJECT_ROOT / "stock_data.db"

if not SOURCE_DB_PATH.exists():
    parent_dir = PROJECT_ROOT.parent
    possible_folders = ["DTB_project", "DeepTrader_baekdoosan", "stock_analysis"]
    
    for folder_name in possible_folders:
        candidate_path = parent_dir / folder_name / "stock_data.db"
        if candidate_path.exists():
            SOURCE_DB_PATH = candidate_path
            print(f"✓ stock_data.db 발견: {candidate_path}")
            break
    
    if not SOURCE_DB_PATH.exists():
        print(f"⚠️ 경고: stock_data.db를 찾을 수 없습니다.")
        sys.exit(1)

                       
CPU_CORES = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

                                    
SEQ_LEN = 60
PRED_LEN = 1
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_DATA_LENGTH = SEQ_LEN + PRED_LEN + 50

# 모든 기술적 지표 리스트 (DB에 저장될 컬럼)
                                                           
INPUT_FEATURES = [
    'close', 'open', 'high', 'low', 'volume', 
    'kospi_close', 'kospi_volume', 'kosdaq_close', 'kosdaq_volume',
    'stochk_14_3_3', 'stochd_14_3_3',  # 원래 이름으로
    'rsi_14', 
    'cci_20_0.015',  # 원래 이름으로
    'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9',
    'obv', 
    'bbl_20_2.0', 'bbm_20_2.0', 'bbu_20_2.0', 'bbb_20_2.0', 'bbp_20_2.0',  # 원래 이름으로
    'mfi_14', 'willr_14', 'adx_14', 'atr_14', 'vwap_d',
    'disparity_5', 'disparity_20', 
    'sma_5', 'sma_20', 'sma_60', 
    'ema_12', 'ema_26', 
    'volume_sma_20'
]

'''
INPUT_FEATURES = [
    'close', 'open', 'high', 'low', 'volume', 
    'kospi_close', 'kospi_volume', 'kosdaq_close', 'kosdaq_volume',
    'rsi_14', 'cci_20', 'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9',
    'mfi_14', 'obv', 'willr_14', 'adx_14', 'atr_14',
    'bbl_20_2', 'bbm_20_2', 'bbu_20_2', 'bbb_20_2', 'bbp_20_2',
    'stochk_14', 'stochd_14',
    'disparity_5', 'disparity_20', 
    'sma_5', 'sma_20', 'sma_60', 
    'ema_12', 'ema_26', 
    'volume_sma_20'
]
'''
TARGET_COLUMN = 'target_close'

# ==================================================
# 2. 시스템 관리
# ==================================================
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"preprocessing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"로깅 시작. 로그 파일: {log_file}")
    return log_file

def setup_directories():
    for directory in [PROCESSED_DATA_DIR, SCALER_DIR, METADATA_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("결과물 저장 디렉토리 설정 완료.")

def get_db_connection(db_path):
    try:
        return sqlite3.connect(db_path, timeout=10)
    except sqlite3.Error as e:
        logging.error(f"DB 연결 오류 ({db_path}): {e}")
        return None

def get_all_stock_info(conn):
       
                                                                              
       
    try:
        query = "SELECT ticker, market FROM stocks WHERE market IN ('KOSPI', 'KOSDAQ')"
        df = pd.read_sql_query(query, conn)
        stock_info = list(zip(df['ticker'], df['market']))
        
                                                                                  
        stock_info.append(('KOSPI', 'INDEX'))
        stock_info.append(('KOSDAQ', 'INDEX'))
        
        logging.info(f"주식 종목 {len(df)}개와 지수 2개를 포함하여 총 {len(stock_info)}개 처리 예정")
        return stock_info
    except Exception as e:
        logging.error(f"전체 티커 목록 조회 오류: {e}")
        return []

def get_data_for_ticker(conn, ticker):
    try:
        query = "SELECT * FROM daily_prices WHERE ticker = ? ORDER BY date"
        df = pd.read_sql_query(query, conn, params=(ticker,))
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logging.error(f"티커 데이터 조회 오류 {ticker}: {e}")
        return pd.DataFrame()

# ==================================================
# 3. 전처리 파이프라인
# ==================================================

def clean_and_prepare_data(df: pd.DataFrame, ticker: str) -> (pd.DataFrame, str):
    if df.empty: 
        return None, "원본 데이터 없음"
    
    if ticker not in ['KOSPI', 'KOSDAQ'] and df['date'].max() < pd.Timestamp.now() - pd.DateOffset(months=6):
        return None, f"6개월 이상 데이터 중단 (최종일: {df['date'].max().date()})"
    
    if ticker == '005930':
        df = df[df['date'] >= pd.to_datetime('2018-05-04')].copy()

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=numeric_cols, inplace=True)
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    initial_rows = len(df)
    df = df[(df['volume'] > 0) & (df['close'] > 0)].copy()
    if len(df) < initial_rows:
        logging.debug(f"[{ticker}] 이상치 제거: {initial_rows} -> {len(df)}")
    
    return df, None

                                                           
        
                                                         
                   
                          
                                                                 
                    
def add_features_and_target(df: pd.DataFrame, ticker: str) -> (pd.DataFrame, list):
    df_ta = df.copy()
    
    # pandas_ta 사용을 위해 datetime index 설정
    df_with_dt_index = df_ta.set_index('date')
    
    # pandas_ta 지표 계산 (safe_add_ta_indicator 사용)
    def safe_add_ta_indicator(df, indicator_name, **kwargs):
        try:
            df.ta(indicator_name, append=True, **kwargs)
            return True
        except Exception as e:
            logging.warning(f"TA 지표 '{indicator_name}' 계산 실패: {e}")
            return False
    
    # 각 지표 계산
    safe_add_ta_indicator(df_with_dt_index, "stoch")  # stochk_14_3_3, stochd_14_3_3
    safe_add_ta_indicator(df_with_dt_index, "rsi")    # rsi_14
    safe_add_ta_indicator(df_with_dt_index, "cci")    # cci_20_0.015
    safe_add_ta_indicator(df_with_dt_index, "macd")   # macd_12_26_9 등
    safe_add_ta_indicator(df_with_dt_index, "bbands") # bbl_20_2.0 등
    safe_add_ta_indicator(df_with_dt_index, "obv")    # obv
    safe_add_ta_indicator(df_with_dt_index, "willr")  # willr_14
    safe_add_ta_indicator(df_with_dt_index, "adx")    # adx_14
    safe_add_ta_indicator(df_with_dt_index, "atr")    # atr_14
    safe_add_ta_indicator(df_with_dt_index, "vwap")   # vwap_d
    
    df_ta = df_with_dt_index.reset_index()
    
    # MFI 수동 계산 (기존 코드와 동일)
    try:
        typical_price = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
        money_flow = typical_price * df_ta['volume']
        delta_tp = typical_price.diff()
        positive_mf = pd.Series(np.where(delta_tp > 0, money_flow, 0), index=df_ta.index)
        negative_mf = pd.Series(np.where(delta_tp < 0, money_flow, 0), index=df_ta.index)
        positive_mf_sum = positive_mf.rolling(window=14, min_periods=1).sum()
        negative_mf_sum = negative_mf.rolling(window=14, min_periods=1).sum()
        money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, 1e-9)
        df_ta['MFI_14'] = 100 - (100 / (1 + money_flow_ratio))
    except Exception as e:
        logging.warning(f"[{ticker}] MFI 수동 계산 실패: {e}")

    # SMA, EMA, Volume SMA, Disparity (기존과 동일)
    for n in [5, 20, 60]:
        try: df_ta[f'SMA_{n}'] = df_ta['close'].rolling(window=n).mean()
        except Exception: pass
    for n in [12, 26]:
        try: df_ta[f'EMA_{n}'] = df_ta['close'].ewm(span=n, adjust=False).mean()
        except Exception: pass
    try: df_ta['VOLUME_SMA_20'] = df_ta['volume'].rolling(window=20).mean()
    except Exception: pass
    for n in [5, 20]:
        try:
            sma = df_ta[f'SMA_{n}']
            df_ta[f'DISPARITY_{n}'] = (df_ta['close'] / sma - 1) * 100
        except Exception: pass

    # 컬럼명 소문자 변환
    df_ta.columns = [col.lower() for col in df_ta.columns]
    
    # Target 설정
    df_ta[TARGET_COLUMN] = df_ta['close'].shift(-PRED_LEN)
    
    final_indicators = [col for col in INPUT_FEATURES if col in df_ta.columns]
    
    logging.debug(f"[{ticker}] 생성된 지표: {len(final_indicators)}개")
    return df_ta, final_indicators

'''    
def add_features_and_target(df: pd.DataFrame, ticker: str) -> (pd.DataFrame, list):
    """모든 기술적 지표를 계산하여 추가"""
    df_ta = df.copy()
    
    if not USE_TA:
        logging.warning(f"[{ticker}] ta 라이브러리 없음 - 기본 지표만 계산")
        # 기본 지표만 계산
        for n in [5, 20, 60]:
            df_ta[f'sma_{n}'] = df_ta['close'].rolling(window=n).mean()
        for n in [12, 26]:
            df_ta[f'ema_{n}'] = df_ta['close'].ewm(span=n, adjust=False).mean()
        df_ta['volume_sma_20'] = df_ta['volume'].rolling(window=20).mean()
        
        # 기본값 설정
        for col in INPUT_FEATURES:
            if col not in df_ta.columns and col not in ['open', 'high', 'low', 'close', 'volume', 'kospi_close', 'kospi_volume', 'kosdaq_close', 'kosdaq_volume']:
                df_ta[col] = 0.0
    else:
        # RSI
        df_ta['rsi_14'] = ta.momentum.rsi(df_ta['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df_ta['close'], window_slow=26, window_fast=12, window_sign=9)
        df_ta['macd_12_26_9'] = macd.macd()
        df_ta['macdh_12_26_9'] = macd.macd_diff()
        df_ta['macds_12_26_9'] = macd.macd_signal()
        
        # CCI (소수점 제거)
        df_ta['cci_20'] = ta.trend.cci(df_ta['high'], df_ta['low'], df_ta['close'], window=20)
        
        # Bollinger Bands (소수점 제거)
        bb = ta.volatility.BollingerBands(df_ta['close'], window=20, window_dev=2)
        df_ta['bbl_20_2'] = bb.bollinger_lband()
        df_ta['bbm_20_2'] = bb.bollinger_mavg()
        df_ta['bbu_20_2'] = bb.bollinger_hband()
        df_ta['bbb_20_2'] = bb.bollinger_wband()
        df_ta['bbp_20_2'] = bb.bollinger_pband()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df_ta['high'], df_ta['low'], df_ta['close'], window=14, smooth_window=3)
        df_ta['stochk_14'] = stoch.stoch()
        df_ta['stochd_14'] = stoch.stoch_signal()
        
        # OBV
        df_ta['obv'] = ta.volume.on_balance_volume(df_ta['close'], df_ta['volume'])
        
        # Williams %R
        df_ta['willr_14'] = ta.momentum.williams_r(df_ta['high'], df_ta['low'], df_ta['close'], lbp=14)
        
        # ADX
        df_ta['adx_14'] = ta.trend.adx(df_ta['high'], df_ta['low'], df_ta['close'], window=14)
        
        # ATR
        df_ta['atr_14'] = ta.volatility.average_true_range(df_ta['high'], df_ta['low'], df_ta['close'], window=14)
        
        # MFI (수동 계산)
        try:
            typical_price = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
            money_flow = typical_price * df_ta['volume']
            delta_tp = typical_price.diff()
            positive_mf = pd.Series(np.where(delta_tp > 0, money_flow, 0), index=df_ta.index)
            negative_mf = pd.Series(np.where(delta_tp < 0, money_flow, 0), index=df_ta.index)
            positive_mf_sum = positive_mf.rolling(window=14, min_periods=1).sum()
            negative_mf_sum = negative_mf.rolling(window=14, min_periods=1).sum()
            money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, 1e-9)
            df_ta['mfi_14'] = 100 - (100 / (1 + money_flow_ratio))
        except Exception as e:
            logging.warning(f"[{ticker}] MFI 계산 실패: {e}")
            df_ta['mfi_14'] = 50.0
        
        # SMA
        for n in [5, 20, 60]:
            df_ta[f'sma_{n}'] = df_ta['close'].rolling(window=n).mean()
        
        # EMA
        for n in [12, 26]:
            df_ta[f'ema_{n}'] = df_ta['close'].ewm(span=n, adjust=False).mean()
        
        # Volume SMA
        df_ta['volume_sma_20'] = df_ta['volume'].rolling(window=20).mean()
        
        # Disparity
        for n in [5, 20]:
            sma = df_ta[f'sma_{n}']
            df_ta[f'disparity_{n}'] = (df_ta['close'] / sma - 1) * 100
    
    # Target 컬럼
    df_ta[TARGET_COLUMN] = df_ta['close'].shift(-PRED_LEN)
                                                         
                                                           
                                                               
                                                         
                                                             
                                                         
                                                         
                                                           
    
    # 최종 피처 리스트 (실제로 있는 컬럼만)

        
                                                                           
                                                    
                                       
                                                                                         
                                                                                         
                                                                             
                                                                             
                                                                                                        
                                                              
                          
                                                                    

                         
                                                                        
                              
                      
                                                                                
                              
                                                                           
                          
                     
            
                                   
                                                                      
                              

                                                          
    final_indicators = [col for col in INPUT_FEATURES if col in df_ta.columns]

                                                          
    
    logging.debug(f"[{ticker}] 생성된 지표: {len(final_indicators)}개")
    return df_ta, final_indicators
'''

def smart_fill_na(df: pd.DataFrame, all_features: list) -> pd.DataFrame:
    df_filled = df.copy()
    for feature in all_features:
        if feature not in df_filled.columns: 
            continue
        
        if 'volume' in feature:
            df_filled[feature] = df_filled[feature].fillna(0)
        elif any(ind in feature for ind in ['rsi', 'stoch', 'cci', 'mfi', 'willr']):
            df_filled[feature] = df_filled[feature].ffill().fillna(50)
        elif any(ind in feature for ind in ['macd', 'obv', 'atr', 'adx']):
            df_filled[feature] = df_filled[feature].ffill().fillna(0)
        elif 'disparity' in feature:
            df_filled[feature] = df_filled[feature].fillna(0)
        else:
            df_filled[feature] = df_filled[feature].ffill().bfill()
            
    return df_filled.dropna(subset=all_features)

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    return df

# ==================================================
# 4. 핵심 워커
# ==================================================
def preprocess_single_ticker(df_stock, df_kospi, df_kosdaq):
       
                                                           
                                                                                       
       
    ticker = df_stock['ticker'].iloc[0]
    df_proc, reason = clean_and_prepare_data(df_stock, ticker)
    if reason: 
        return None, [], reason

    # 지수 데이터 병합
    for df_index, name in [(df_kospi, 'KOSPI'), (df_kosdaq, 'KOSDAQ')]:
        if not df_index.empty and ticker != name:
            rename_map = {'close': f'{name.lower()}_close', 'volume': f'{name.lower()}_volume'}
            df_index_renamed = df_index[['date'] + list(rename_map.keys())].rename(columns=rename_map)
            df_proc = pd.merge(df_proc, df_index_renamed, on='date', how='left')
            
                                                                                      
            for col_name in rename_map.values():
                if col_name in df_proc.columns:
                    df_proc[col_name] = df_proc[col_name].ffill()

    df_proc, generated_features = add_features_and_target(df_proc, ticker)
    
    all_cols_to_process = generated_features + [TARGET_COLUMN]
    df_proc = smart_fill_na(df_proc, all_cols_to_process)
    
    # 로그 변환
    log_cols = [c for c in ['open', 'high', 'low', 'close', 'kospi_close', 'kosdaq_close', TARGET_COLUMN] if c in df_proc.columns]
    for col in log_cols:
        df_proc.loc[df_proc[col] > 0, col] = np.log1p(df_proc.loc[df_proc[col] > 0, col])
    
    # 최종 컬럼 선택
    final_cols = ['date', 'ticker'] + generated_features + [TARGET_COLUMN]
    df_proc = df_proc.reindex(columns=final_cols).dropna()
    df_proc = optimize_dtypes(df_proc)
    
    return df_proc, generated_features, None

                                                                

def validate_and_save(df: pd.DataFrame, features: list, ticker: str, market: str):
    if len(df) < MIN_DATA_LENGTH:
        return None, f"데이터 길이 부족 ({len(df)} < {MIN_DATA_LENGTH})"
    if not features:
        return None, "생성된 피처가 없음"
    if df[features].isnull().sum().sum() > 0 or np.isinf(df[features].values).any():
        return None, "최종 데이터에 결측치/무한값 존재"
    
    try:
        total_len = len(df)
        
                                                 
        test_size = int(total_len * TEST_RATIO)
        val_size = int(total_len * VAL_RATIO)
        train_size = total_len - val_size - test_size
        
        scaler = StandardScaler()
                                                                                                       
        scaler.fit(df.iloc[:train_size][features])
        joblib.dump(scaler, SCALER_DIR / f"{ticker}.pkl")
        
        metadata = {
            'ticker': ticker, 'market': market, 'version': 'v9_fixed',
            'timestamp': datetime.now().isoformat(),
            'data_shape': {'rows': len(df), 'features': len(features)},
            'date_range': {
                'start': df['date'].iloc[0].strftime('%Y-%m-%d'), 
                'end': df['date'].iloc[-1].strftime('%Y-%m-%d')
            },
            'features': features,
            'target': TARGET_COLUMN,
                                                                                         
            'split_sizes': {
                'train': train_size,
                'validation': val_size,
                'test': test_size
            }
        }
        
        with open(METADATA_DIR / f"{ticker}.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        return df, None
        
    except Exception as e:
        return None, f"아티팩트 저장 오류: {e}"
        

def worker(args):
    ticker, market, source_db_path, df_kospi, df_kosdaq = args
    try:
        with get_db_connection(source_db_path) as conn:
            df_stock = get_data_for_ticker(conn, ticker)
        
        processed_df, generated_features, reason = preprocess_single_ticker(df_stock, df_kospi, df_kosdaq)
        if reason: 
            return {'status': 'fail', 'ticker': ticker, 'market': market, 'reason': reason}
        
        final_df, reason = validate_and_save(processed_df, generated_features, ticker, market)
        if reason: 
            return {'status': 'fail', 'ticker': ticker, 'market': market, 'reason': reason}
            
        return {
            'status': 'success', 
            'ticker': ticker, 
            'market': market, 
            'data': final_df, 
            'features': generated_features
        }
        
    except Exception as e:
        logging.error(f"[{ticker}] 워커 예외 발생: {e}", exc_info=True)
        return {'status': 'fail', 'ticker': ticker, 'market': market, 'reason': f"예외 발생: {e}"}

# ==================================================
# 5. 메인 실행
# ==================================================
def main():
    start_time = datetime.now()
    log_file = setup_logging()
    setup_directories()
    
    if DESTINATION_DB_PATH.exists():
        # 타임스탬프가 포함된 백업 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = DESTINATION_DB_PATH.with_name(f"processed_stock_data_{timestamp}.db.backup")
        DESTINATION_DB_PATH.rename(backup_path)
        logging.info(f"기존 DB 백업: '{backup_path}'")
    
    try:
        with get_db_connection(SOURCE_DB_PATH) as conn:
            if not conn: 
                return
            all_stock_info = get_all_stock_info(conn)
            df_kospi = get_data_for_ticker(conn, 'KOSPI')
            df_kosdaq = get_data_for_ticker(conn, 'KOSDAQ')

        for df_index in [df_kospi, df_kosdaq]:
            if not df_index.empty:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df_index.columns:
                        df_index[col] = pd.to_numeric(df_index[col], errors='coerce').astype(float)
                df_index.dropna(subset=['close', 'volume'], inplace=True)

        tasks = [(ticker, market, SOURCE_DB_PATH, df_kospi, df_kosdaq) for ticker, market in all_stock_info]
        logging.info(f"총 {len(tasks)}개 종목 및 지수 전처리 시작 (CPU 코어: {CPU_CORES}개)...")

        all_results = []
        with mp.Pool(processes=CPU_CORES) as pool:
            results_iter = pool.imap_unordered(worker, tasks)
            all_results = list(tqdm(results_iter, total=len(tasks), desc="종목 및 지수 전처리"))
        
        success_results = [r for r in all_results if r and r['status'] == 'success']
        failed_results = [r for r in all_results if r and r['status'] == 'fail']

        if success_results:
            combined_df = pd.concat([res['data'] for res in success_results], ignore_index=True)
            logging.info(f"총 {len(combined_df)} 행 데이터 DB 저장 시작...")
            
            with get_db_connection(DESTINATION_DB_PATH) as conn_dest:
                combined_df.to_sql('processed_daily_prices', conn_dest, if_exists='replace', index=False, chunksize=10000)
                conn_dest.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON processed_daily_prices (ticker, date);")
                conn_dest.commit()
            
            logging.info("DB 저장 및 인덱싱 완료.")
            del combined_df
            gc.collect()
        
        save_summary_report(success_results, failed_results, start_time, log_file)

    except Exception as e:
        logging.error("메인 프로세스 오류 발생!", exc_info=True)
    finally:
        logging.info(f"총 처리 시간: {datetime.now() - start_time}")
        logging.info("="*20 + " 전처리 파이프라인(v9_fixed) 종료 " + "="*20)

def save_summary_report(success, failed, start_time, log_file):
    market_dist = pd.Series([r['market'] for r in success]).value_counts().to_dict()
    failure_reasons = pd.Series([r['reason'].split('(')[0].strip() for r in failed]).value_counts().to_dict()

    summary = {
        'run_info': {
            'version': 'v9_fixed', 
            'start_time': start_time.isoformat(),
            'duration': str(datetime.now() - start_time), 
            'log_file': str(log_file)
        },
        'results': {
            'total_processed': len(success) + len(failed),
            'success_count': len(success), 
            'failed_count': len(failed),
            'success_rate': f"{len(success) / (len(success) + len(failed)) * 100 if success or failed else 0:.2f}%"
        },
        'data_summary': {
            'market_distribution': market_dist
        },
        'failure_analysis': failure_reasons
    }
    
    summary_path = PROCESSED_DATA_DIR / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    logging.info(f"최종 요약 리포트 저장: {summary_path}")
    print(json.dumps(summary, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    mp.freeze_support()
    main()