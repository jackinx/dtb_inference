# FEDformer 데이터 전처리 스크립트 (v9 - 지수 데이터 포함)
#
# 주요 개선 사항 (v8 기반):
# 1. [핵심] 처리 대상에 'KOSPI', 'KOSDAQ' 지수를 포함하여 일반 종목과 동일한 방식으로 전처리
# 2. [정확성] 지수 데이터 처리 시, 자기 자신을 피처로 병합하지 않도록 로직 수정

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

# ==================================================
# 1. 설정 (Configuration)
# ==================================================
# --- 경로 설정 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

# 경로를 v9에 맞게 수정
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data_v9"
DESTINATION_DB_PATH = PROCESSED_DATA_DIR / "processed_stock_data.db"
SCALER_DIR = PROCESSED_DATA_DIR / "scalers"
METADATA_DIR = PROCESSED_DATA_DIR / "metadata"
LOG_DIR = PROCESSED_DATA_DIR / "logs"

# 소스 DB 경로는 기존 경로를 사용
SOURCE_PROJECT_ROOT = PROJECT_ROOT.parent / "DeepTrader_baekdoosan"
if not SOURCE_PROJECT_ROOT.exists():
    SOURCE_PROJECT_ROOT = Path(r"C:\Users\jacki\OneDrive\Documents\anaconda_projects\DeepTrader_baekdoosan")
SOURCE_DB_PATH = SOURCE_PROJECT_ROOT / "stock_data.db"


# --- 처리 설정 ---
CPU_CORES = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

# --- 모델 학습 파라미터 ---
SEQ_LEN = 60
PRED_LEN = 1
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_DATA_LENGTH = SEQ_LEN + PRED_LEN + 50


# INPUT_FEATURES 리스트는 마스터 목록으로 사용
INPUT_FEATURES = [
    'close', 'open', 'high', 'low', 'volume', 'kospi_close', 'kospi_volume', 'kosdaq_close', 'kosdaq_volume',
    'stochk_14_3_3', 'stochd_14_3_3', 'rsi_14', 'cci_20_0.015', 'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9',
    'obv', 'bbl_20_2.0', 'bbm_20_2.0', 'bbu_20_2.0', 'bbb_20_2.0', 'bbp_20_2.0', 'mfi_14', 'willr_14', 'adx_14',
    'atr_14', 'vwap_d', 'disparity_5', 'disparity_20', 'sma_5', 'sma_20', 'sma_60', 'ema_12', 'ema_26', 'volume_sma_20'
]
TARGET_COLUMN = 'target_close'

# ==================================================
# 2. 시스템 관리 (로깅, 디렉토리, DB)
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
    """
    [수정] KOSPI와 KOSDAQ 지수를 처리 목록에 명시적으로 추가
    """
    try:
        query = "SELECT ticker, market FROM stocks WHERE market IN ('KOSPI', 'KOSDAQ')"
        df = pd.read_sql_query(query, conn)
        stock_info = list(zip(df['ticker'], df['market']))
        
        # 지수를 처리 목록에 추가. 시장명은 'INDEX'로 임의 지정
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
# 3. 전처리 파이프라인 함수
# ==================================================

def clean_and_prepare_data(df: pd.DataFrame, ticker: str) -> (pd.DataFrame, str):
    if df.empty: return None, "원본 데이터 없음"
    # 지수는 데이터 중단 체크 제외
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

def safe_add_ta_indicator(df, ta_function, name, **kwargs):
    try:
        df.ta(handler=ta_function, append=True, **kwargs)
        return True
    except Exception as e:
        logging.warning(f"TA 지표 '{name}' 계산 실패: {e}")
        return False

def add_features_and_target(df: pd.DataFrame, ticker: str) -> (pd.DataFrame, list):
    df_ta = df.copy()
    
    df_with_dt_index = df_ta.set_index('date')
    
    safe_add_ta_indicator(df_with_dt_index, "stoch", "STOCH")
    safe_add_ta_indicator(df_with_dt_index, "rsi", "RSI")
    safe_add_ta_indicator(df_with_dt_index, "cci", "CCI")
    safe_add_ta_indicator(df_with_dt_index, "macd", "MACD")
    safe_add_ta_indicator(df_with_dt_index, "bbands", "BBANDS")
    safe_add_ta_indicator(df_with_dt_index, "obv", "OBV")
    safe_add_ta_indicator(df_with_dt_index, "willr", "WILLR")
    safe_add_ta_indicator(df_with_dt_index, "adx", "ADX")
    safe_add_ta_indicator(df_with_dt_index, "atr", "ATR")
    safe_add_ta_indicator(df_with_dt_index, "vwap", "VWAP")
    
    df_ta = df_with_dt_index.reset_index()

    try:
        typical_price = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
        money_flow = typical_price * df_ta['volume']
        delta_tp = typical_price.diff()
        positive_mf = pd.Series(np.where(delta_tp > 0, money_flow, 0), index=df_ta.index)
        negative_mf = pd.Series(np.where(delta_tp < 0, money_flow, 0), index=df_ta.index)
        positive_mf_sum = positive_mf.rolling(window=14, min_periods=1).sum()
        negative_mf_sum = negative_mf.rolling(window=14, min_periods=1).sum()
        money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, 1e-9) # 0으로 나누기 방지
        df_ta['MFI_14'] = 100 - (100 / (1 + money_flow_ratio))
    except Exception as e:
        logging.warning(f"[{ticker}] MFI 수동 계산 실패: {e}")

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

    df_ta.columns = [col.lower() for col in df_ta.columns]
    final_indicators = [col for col in INPUT_FEATURES if col in df_ta.columns]

    df_ta[TARGET_COLUMN] = df_ta['close'].shift(-PRED_LEN)
    
    logging.debug(f"[{ticker}] 생성된 지표: {len(final_indicators)}개")
    return df_ta, final_indicators

def smart_fill_na(df: pd.DataFrame, all_features: list) -> pd.DataFrame:
    df_filled = df.copy()
    for feature in all_features:
        if feature not in df_filled.columns: continue
        
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
# 4. 핵심 워커 및 검증/저장 로직
# ==================================================
def preprocess_single_ticker(df_stock, df_kospi, df_kosdaq):
    """
    [핵심 수정] 단일 종목 전처리 파이프라인
    - 현재 티커가 지수 자신일 경우, 피처로 병합하지 않도록 수정
    """
    ticker = df_stock['ticker'].iloc[0]
    df_proc, reason = clean_and_prepare_data(df_stock, ticker)
    if reason: return None, [], reason

    # [수정] 현재 처리 중인 티커가 지수 자신일 경우, 피처로 병합하지 않도록 조건 추가
    for df_index, name in [(df_kospi, 'KOSPI'), (df_kosdaq, 'KOSDAQ')]:
        if not df_index.empty and ticker != name: # 이 조건이 핵심
            rename_map = {'close': f'{name.lower()}_close', 'volume': f'{name.lower()}_volume'}
            df_index_renamed = df_index[['date'] + list(rename_map.keys())].rename(columns=rename_map)
            df_proc = pd.merge(df_proc, df_index_renamed, on='date', how='left')
            
            # 피처 병합 후 발생할 수 있는 NaN은 앞선 값으로 채워줌
            for col_name in rename_map.values():
                if col_name in df_proc.columns:
                    df_proc[col_name] = df_proc[col_name].ffill()

    df_proc, generated_features = add_features_and_target(df_proc, ticker)
    
    all_cols_to_process = generated_features + [TARGET_COLUMN]
    df_proc = smart_fill_na(df_proc, all_cols_to_process)
    
    # 지수 자신을 피처로 사용하지 않으므로, 로그 변환 대상에서도 제외될 수 있도록 방어
    log_cols = [c for c in ['open', 'high', 'low', 'close', 'kospi_close', 'kosdaq_close', TARGET_COLUMN] if c in df_proc.columns]
    for col in log_cols:
        df_proc.loc[df_proc[col] > 0, col] = np.log1p(df_proc.loc[df_proc[col] > 0, col])
        
    final_cols = ['date', 'ticker'] + generated_features + [TARGET_COLUMN]
    df_proc = df_proc.reindex(columns=final_cols).dropna()
    df_proc = optimize_dtypes(df_proc)
    
    return df_proc, generated_features, None

# preprocess_v9_with_index.py의 validate_and_save 함수 수정

def validate_and_save(df: pd.DataFrame, features: list, ticker: str, market: str):
    if len(df) < MIN_DATA_LENGTH:
        return None, f"데이터 길이 부족 ({len(df)} < {MIN_DATA_LENGTH})"
    if not features:
        return None, "생성된 피처가 없음"
    if df[features].isnull().sum().sum() > 0 or np.isinf(df[features].values).any():
        return None, "최종 데이터에 결측치/무한값 존재"
    try:
        total_len = len(df)
        
        # [수정] 데이터 분할 크기 계산
        test_size = int(total_len * TEST_RATIO)
        val_size = int(total_len * VAL_RATIO)
        train_size = total_len - val_size - test_size
        
        scaler = StandardScaler()
        # 주의: train_size-1이 아닌 train_size까지 사용해야 올바른 슬라이싱입니다.
        scaler.fit(df.loc[:train_size-1, features])
        joblib.dump(scaler, SCALER_DIR / f"{ticker}.pkl")
        
        metadata = {
            'ticker': ticker, 'market': market, 'version': 'v9',
            'timestamp': datetime.now().isoformat(),
            'data_shape': {'rows': len(df), 'features': len(features)},
            'date_range': {'start': df['date'].iloc[0].strftime('%Y-%m-%d'), 'end': df['date'].iloc[-1].strftime('%Y-%m-%d')},
            'features': features,
            'target': TARGET_COLUMN,
            # [핵심 추가] 데이터 분할 크기 정보를 메타데이터에 저장
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
        if reason: return {'status': 'fail', 'ticker': ticker, 'market': market, 'reason': reason}
        
        final_df, reason = validate_and_save(processed_df, generated_features, ticker, market)
        if reason: return {'status': 'fail', 'ticker': ticker, 'market': market, 'reason': reason}
            
        return {'status': 'success', 'ticker': ticker, 'market': market, 'data': final_df, 'features': generated_features}
    except Exception as e:
        logging.error(f"[{ticker}] 워커 예외 발생: {e}", exc_info=True)
        return {'status': 'fail', 'ticker': ticker, 'market': market, 'reason': f"예외 발생: {e}"}

# ==================================================
# 5. 메인 실행 로직
# ==================================================
def main():
    start_time = datetime.now()
    log_file = setup_logging()
    setup_directories()
    
    if DESTINATION_DB_PATH.exists():
        DESTINATION_DB_PATH.unlink()
        logging.info(f"기존 DB 파일 '{DESTINATION_DB_PATH}' 삭제.")

    try:
        with get_db_connection(SOURCE_DB_PATH) as conn:
            if not conn: return
            all_stock_info = get_all_stock_info(conn) # 수정된 함수 호출
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
            del combined_df; gc.collect()
        
        save_summary_report(success_results, failed_results, start_time, log_file)

    except Exception as e:
        logging.error("메인 프로세스 오류 발생!", exc_info=True)
    finally:
        logging.info(f"총 처리 시간: {datetime.now() - start_time}")
        logging.info("="*20 + " 전처리 파이프라인(v9) 종료 " + "="*20)

def save_summary_report(success, failed, start_time, log_file):
    market_dist = pd.Series([r['market'] for r in success]).value_counts().to_dict()
    failure_reasons = pd.Series([r['reason'].split('(')[0].strip() for r in failed]).value_counts().to_dict()

    summary = {
        'run_info': {
            'version': 'v9', 'start_time': start_time.isoformat(),
            'duration': str(datetime.now() - start_time), 'log_file': str(log_file)
        },
        'results': {
            'total_processed': len(success) + len(failed),
            'success_count': len(success), 'failed_count': len(failed),
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