import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import concurrent.futures
import sys
from urllib.parse import quote

# --- 설정값 ---
DB_FILE = "stock_data.db"
STOCK_LIST_FILE = "stock_list.txt"
SECTOR_MAPPING_FILE = "sector_mapping.txt"  # 추가
CRAWL_START_DATE = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
HEADERS = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
MAX_PAGES_PER_STOCK = 1000

# --- DB 설정 ---
def setup_database():
    """데이터베이스와 테이블을 초기 설정합니다."""
    print(f"'{DB_FILE}' 데이터베이스 설정 중...")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT PRIMARY KEY, 
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            stock_type TEXT,
            sector TEXT)''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker TEXT, 
            date TEXT, 
            open INTEGER, 
            high INTEGER,
            low INTEGER, 
            close INTEGER, 
            volume INTEGER,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker))''')
    
    # 기존 테이블에 sector 컬럼 추가 (없는 경우에만)
    try:
        cur.execute("SELECT sector FROM stocks LIMIT 1")
    except sqlite3.OperationalError:
        print("'sector' 컬럼 추가 중...")
        cur.execute("ALTER TABLE stocks ADD COLUMN sector TEXT")
        print("'sector' 컬럼 추가 완료.")
    
    conn.commit()
    print("데이터베이스 설정 완료.")
    return conn

# --- 파일에서 종목 리스트 읽기 ---
def read_stock_list_from_file(filename):
    """텍스트 파일에서 종목 리스트를 읽어와 딕셔너리 리스트로 반환합니다."""
    print(f"'{filename}' 파일에서 종목 리스트를 읽는 중...")
    tickers_info = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split(',')
                if len(parts) == 3:
                    ticker, name, market = [p.strip() for p in parts]
                    stock_type = 'ETF' if any(keyword in name.upper() for keyword in ['ETF', 'TIGER', 'KODEX']) else '일반'
                    tickers_info.append({
                        'ticker': ticker,
                        'name': name,
                        'market': market,
                        'stock_type': stock_type
                    })
    except FileNotFoundError:
        print(f"오류: '{filename}'을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류: '{filename}' 파일 읽기 중 오류 발생 - {e}")
        return None
    
    print(f"총 {len(tickers_info)}개 종목 정보를 파일에서 읽었습니다.")
    return tickers_info

# --- 섹터 매핑 파일 읽기 (새로 추가) ---
def load_sector_mapping(filename=SECTOR_MAPPING_FILE):
    """섹터 매핑 파일을 읽어서 딕셔너리로 반환합니다."""
    sector_map = {}
    
    if not os.path.exists(filename):
        print(f"경고: '{filename}' 파일을 찾을 수 없습니다. 섹터 정보 없이 진행합니다.")
        return sector_map
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 3:
                    ticker = parts[0].strip()
                    sector = parts[2].strip()
                    sector_map[ticker] = sector
        
        print(f"'{filename}'에서 {len(sector_map)}개 종목의 섹터 정보를 로드했습니다.")
        
    except Exception as e:
        print(f"경고: '{filename}' 파일 읽기 중 오류 발생 - {e}")
    
    return sector_map

# --- 종목 메타데이터 업데이트 (수정) ---
def update_stock_metadata(conn, tickers, sector_map):
    """종목 메타데이터와 섹터 정보를 업데이트합니다."""
    print("\n종목 정보 메타데이터 업데이트 중...")
    cur = conn.cursor()
    
    updated_count = 0
    
    for ticker_info in tqdm(tickers, desc="메타데이터 업데이트"):
        ticker = ticker_info['ticker']
        
        # 섹터 정보 설정
        if ticker_info.get('stock_type') == 'Index':
            ticker_info['sector'] = 'Index'
        elif ticker in sector_map:
            ticker_info['sector'] = sector_map[ticker]
            updated_count += 1
        else:
            # 섹터 매핑에 없으면 기존 DB 값 유지 또는 Unknown
            cur.execute("SELECT sector FROM stocks WHERE ticker = ?", (ticker,))
            existing = cur.fetchone()
            if existing and existing[0]:
                ticker_info['sector'] = existing[0]
            else:
                ticker_info['sector'] = 'Unknown'
    
    # 데이터베이스에 일괄 업데이트
    cur.executemany('''
        INSERT INTO stocks (ticker, name, market, stock_type, sector) 
        VALUES (:ticker, :name, :market, :stock_type, :sector)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name, 
            market=excluded.market, 
            stock_type=excluded.stock_type,
            sector=excluded.sector
    ''', tickers)
    
    conn.commit()
    print(f"{len(tickers)}개 종목 정보 업데이트 완료. (섹터 정보: {updated_count}개)")

# --- 최신 거래일 확인 함수 ---
def get_latest_market_day():
    """네이버 금융에서 KOSPI 지수의 가장 최신 거래일을 조회하여 반환합니다."""
    try:
        url = "https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI"
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content.decode('euc-kr', 'replace'), 'html.parser')
        latest_date_str = soup.select_one('table.type_1 td.date').get_text(strip=True).replace('.', '-')
        return latest_date_str
    except Exception as e:
        print(f"경고: 최신 거래일 조회 실패. 어제 날짜를 기준으로 합니다. ({e})")
        return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

# --- 병렬 처리 워커 함수 (개별 종목용) ---
def fetch_prices_for_worker(job_info):
    worker_id, ticker, last_date_in_db, stock_name = job_info
    page = 1
    all_price_data = []
    status_line = worker_id + 6

    while page <= MAX_PAGES_PER_STOCK:
        print(f"\033[{status_line};0H\033[K  [Worker {worker_id+1:02d}] {stock_name[:10]}({ticker}) -> Page {page} 처리 중...", end="", flush=True)

        try:
            url = f"https://finance.naver.com/item/sise_day.naver?code={ticker}&page={page}"
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\033[{status_line};0H\033[K  [Worker {worker_id+1:02d}] {stock_name[:10]}({ticker}) -> 요청 실패 ({e}). 중단.", end="", flush=True)
            break
            
        soup = BeautifulSoup(response.content.decode('euc-kr', 'replace'), 'html.parser')
        
        rows = soup.select('table.type2 tr[onmouseover]')
        if not rows:
            break

        stop_processing = False
        for row in rows:
            cols = row.find_all('td')
            if len(cols) != 7: continue

            try:
                date_str = cols[0].get_text(strip=True).replace('.', '-')
                if not date_str: continue
                
                if last_date_in_db and date_str <= last_date_in_db:
                    stop_processing = True
                    break
                
                if not last_date_in_db and date_str < CRAWL_START_DATE:
                    stop_processing = True
                    break

                all_price_data.append({
                    'ticker': ticker, 'date': date_str,
                    'open': int(cols[3].get_text(strip=True).replace(',', '')),
                    'high': int(cols[4].get_text(strip=True).replace(',', '')),
                    'low': int(cols[5].get_text(strip=True).replace(',', '')),
                    'close': int(cols[1].get_text(strip=True).replace(',', '')),
                    'volume': int(cols[6].get_text(strip=True).replace(',', ''))
                })
            except (ValueError, IndexError):
                continue
        
        if stop_processing:
            break

        page += 1
        time.sleep(random.uniform(0.1, 0.3))

    print(f"\033[{status_line};0H\033[K", end="", flush=True)
    return ticker, all_price_data

# --- 지수 데이터 수집 함수 ---
def fetch_index_prices(index_info):
    """지수 데이터를 수집하는 전용 함수"""
    index_code, last_date_in_db = index_info
    print(f"\n{index_code} 지수 데이터 수집 시작... (DB 마지막 날짜: {last_date_in_db or '없음'})")
    
    page = 1
    all_price_data = []
    
    while page <= MAX_PAGES_PER_STOCK:
        try:
            url = f"https://finance.naver.com/sise/sise_index_day.naver?code={index_code}&page={page}"
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  -> {index_code}: 페이지 {page} 요청 실패 ({e}). 중단.")
            break
            
        soup = BeautifulSoup(response.content.decode('euc-kr', 'replace'), 'html.parser')
        
        rows = soup.select('table.type_1 tr')
        if not soup.select_one('td.date'):
            break
        
        stop_processing = False
        for row in rows:
            cols = row.find_all('td')
            if len(cols) != 6: 
                continue

            try:
                date_str = cols[0].get_text(strip=True).replace('.', '-')
                if not date_str: continue

                if last_date_in_db and date_str <= last_date_in_db:
                    stop_processing = True
                    break
                
                if not last_date_in_db and date_str < CRAWL_START_DATE:
                    stop_processing = True
                    break

                close = float(cols[1].get_text(strip=True).replace(',', ''))
                volume_in_thousands = int(cols[4].get_text(strip=True).replace(',', ''))

                all_price_data.append({
                    'ticker': index_code, 'date': date_str,
                    'open': int(close), 'high': int(close), 'low': int(close), 'close': int(close),
                    'volume': volume_in_thousands * 1000
                })
            except (ValueError, IndexError):
                continue
        
        if stop_processing:
            break
        
        page += 1
        time.sleep(random.uniform(0.2, 0.5))
        
    return all_price_data

# --- 메인 실행 로직 (수정) ---
def main():
    conn = setup_database()
    
    all_tickers = read_stock_list_from_file(STOCK_LIST_FILE)
    if not all_tickers:
        conn.close()
        return

    # 섹터 매핑 파일 로드 (추가)
    sector_map = load_sector_mapping()
    
    # 메타데이터 업데이트 (섹터 정보 포함)
    update_stock_metadata(conn, all_tickers, sector_map)
    
    jobs = []
    skipped_count = 0
    cur = conn.cursor()
    
    latest_market_day_str = get_latest_market_day()
    print(f"\n최신 거래일({latest_market_day_str})까지 데이터 업데이트가 필요한 종목을 찾습니다...")

    for stock in tqdm(all_tickers, desc="작업 목록 생성"):
        ticker = stock['ticker']
        cur.execute("SELECT MAX(date) FROM daily_prices WHERE ticker = ?", (ticker,))
        last_date_in_db = cur.fetchone()[0]
        
        if not last_date_in_db or last_date_in_db < latest_market_day_str:
            jobs.append((ticker, last_date_in_db, stock['name']))
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"{skipped_count}개 종목은 이미 최신 상태이므로 건너뜁니다.")

    # 개별 종목 병렬 처리
    if jobs:
        max_workers = min((os.cpu_count() or 1) * 2, len(jobs))
        print(f"\n총 {len(jobs)}개 종목의 시세 데이터 병렬 업데이트 시작... (최대 {max_workers}개 동시 처리)")
        
        jobs_with_id = [ (i % max_workers, *job) for i, job in enumerate(jobs) ]
        print("\n" * (max_workers + 2))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results_iterator = executor.map(fetch_prices_for_worker, jobs_with_id)
                
                for ticker, price_data in tqdm(results_iterator, total=len(jobs), desc="[전체 진행률]"):
                    if price_data:
                        try:
                            cur.executemany('''
                                INSERT OR IGNORE INTO daily_prices 
                                (ticker, date, open, high, low, close, volume) 
                                VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
                            ''', price_data)
                            conn.commit()
                        except Exception as e:
                            print(f"\n오류: {ticker} 데이터 저장 중 문제 발생 - {e}")
        except KeyboardInterrupt:
            print("\n\n사용자 요청으로 작업을 중단합니다...")
            sys.exit(0)
    else:
        print("\n새로 업데이트할 개별 종목이 없습니다.")

    print("\n\n개별 종목 업데이트 완료.")
    
    # --- 지수 데이터 업데이트 ---
    print("\n--- 지수 데이터 업데이트 시작 ---")
    index_list = [
        {'ticker': 'KOSPI', 'name': '코스피'},
        {'ticker': 'KOSDAQ', 'name': '코스닥'}
    ]
    update_stock_metadata(conn, [{'ticker': i['ticker'], 'name': i['name'], 'market': 'Index', 'stock_type': 'Index'} for i in index_list], sector_map)

    for index in index_list:
        index_code = index['ticker']
        cur.execute("SELECT MAX(date) FROM daily_prices WHERE ticker = ?", (index_code,))
        last_date_in_db = cur.fetchone()[0]
        
        if not last_date_in_db or last_date_in_db < latest_market_day_str:
            price_data = fetch_index_prices((index_code, last_date_in_db))
            if price_data:
                cur.executemany('INSERT OR IGNORE INTO daily_prices VALUES (:ticker, :date, :open, :high, :low, :close, :volume)', price_data)
                conn.commit()
                print(f"-> {index_code} 지수 업데이트 완료. {len(price_data)}개 데이터 추가.")
        else:
            print(f"-> {index_code} 지수는 이미 최신 상태입니다.")

    conn.close()
    print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")

if __name__ == "__main__":
    main()