import urllib.request
import json
import csv
import time
import sqlite3
from datetime import datetime
import concurrent.futures

DB_NAME = "moex_market_data.db"
FUTURES_CSV = "moex_futures_ohlc.csv"
OPTIONS_CSV = "moex_options_ohlc.csv"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS options_ohlc (
            tradedate TEXT,
            secid TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            numtrades INTEGER,
            value REAL,
            PRIMARY KEY (tradedate, secid)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS futures_ohlc (
            tradedate TEXT,
            secid TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            numtrades INTEGER,
            value REAL,
            settleprice REAL,
            PRIMARY KEY (tradedate, secid)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_months (
            market TEXT,
            month_str TEXT,
            PRIMARY KEY (market, month_str)
        )
    """)
    conn.commit()
    conn.close()

def generate_month_tuples(start_year=2023, end_year=2026, end_month=9):
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_month:
                break
            months.append((y, m))
    return months

def fetch_market_month(market, ym_tuple, retries=5):
    year, month = ym_tuple
    month_str = f"{year:04d}-{month:02d}"

    start_d = f"{year:04d}-{month:02d}-01"
    if month == 12:
        end_d = f"{year:04d}-12-31"
    else:
        if month in (1, 3, 5, 7, 8, 10, 12):
            last_day = 31
        elif month in (4, 6, 9, 11):
            last_day = 30
        else:
            last_day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
        end_d = f"{year:04d}-{month:02d}-{last_day:02d}"

    if year == 2026 and month == 9:
        end_d = "2026-09-08"

    records = []
    start = 0
    while True:
        url = f"https://iss.moex.com/iss/history/engines/futures/markets/{market}/securities.json?from={start_d}&till={end_d}&numtrades=1&start={start}"

        data = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    break
            except Exception as e:
                time.sleep(0.5 * (attempt + 1))

        if not data or 'history' not in data:
            print(f"Warning: Failed to fetch {market} month {month_str} at start={start}")
            break

        cols = data['history']['columns']
        sec_idx = cols.index('SECID')
        date_idx = cols.index('TRADEDATE')
        open_idx = cols.index('OPEN')
        high_idx = cols.index('HIGH')
        low_idx = cols.index('LOW')
        close_idx = cols.index('CLOSE')
        vol_idx = cols.index('VOLUME')
        numtrades_idx = cols.index('NUMTRADES')
        val_idx = cols.index('VALUE') if 'VALUE' in cols else -1
        settle_idx = cols.index('SETTLEPRICE') if 'SETTLEPRICE' in cols else -1

        rows = data['history']['data']
        if not rows:
            break

        for r in rows:
            secid = r[sec_idx]
            tradedate = r[date_idx]

            if tradedate > '2026-09-08' or tradedate < '2023-01-01':
                continue

            open_p = r[open_idx]
            close_p = r[close_idx]
            numtrades = r[numtrades_idx] or 0

            if open_p is not None and close_p is not None and numtrades > 0:
                rec = [
                    tradedate,
                    secid,
                    open_p,
                    r[high_idx],
                    r[low_idx],
                    close_p,
                    r[vol_idx] or 0,
                    numtrades,
                    r[val_idx] if val_idx != -1 else None
                ]
                if market == 'forts':
                    rec.append(r[settle_idx] if settle_idx != -1 else None)
                records.append(tuple(rec))

        start += len(rows)
        cursor = data.get('history.cursor', {}).get('data', [])
        if not cursor or start >= cursor[0][1]:
            break

    return market, month_str, records

def save_to_db(market, month_str, records):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if records:
        if market == 'options':
            cursor.executemany("""
                INSERT OR REPLACE INTO options_ohlc
                (tradedate, secid, open, high, low, close, volume, numtrades, value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
        else:
            cursor.executemany("""
                INSERT OR REPLACE INTO futures_ohlc
                (tradedate, secid, open, high, low, close, volume, numtrades, value, settleprice)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

    cursor.execute("INSERT OR REPLACE INTO processed_months VALUES (?, ?)", (market, month_str))
    conn.commit()
    conn.close()

def export_to_csv():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT tradedate, secid, open, high, low, close, volume, numtrades, value FROM options_ohlc ORDER BY tradedate, secid")
    opt_rows = cursor.fetchall()
    with open(OPTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tradedate", "secid", "open", "high", "low", "close", "volume", "numtrades", "value"])
        writer.writerows(opt_rows)
    print(f"Exported {len(opt_rows)} option records to {OPTIONS_CSV}")

    cursor.execute("SELECT tradedate, secid, open, high, low, close, volume, numtrades, value, settleprice FROM futures_ohlc ORDER BY tradedate, secid")
    fut_rows = cursor.fetchall()
    with open(FUTURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tradedate", "secid", "open", "high", "low", "close", "volume", "numtrades", "value", "settleprice"])
        writer.writerows(fut_rows)
    print(f"Exported {len(fut_rows)} futures records to {FUTURES_CSV}")

    conn.close()

def main():
    init_db()
    months = generate_month_tuples()
    print(f"Total months to fetch per market: {len(months)}")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT market, month_str FROM processed_months")
    processed = set((r[0], r[1]) for r in cursor.fetchall())
    conn.close()

    tasks = []
    for market in ['options', 'forts']:
        for m in months:
            m_str = f"{m[0]:04d}-{m[1]:02d}"
            if (market, m_str) not in processed:
                tasks.append((market, m))

    print(f"Market-months remaining to fetch: {len(tasks)}")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(fetch_market_month, market, m): (market, m) for market, m in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            market, month_str, records = future.result()
            save_to_db(market, month_str, records)
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            rem = (len(tasks) - completed) / rate if rate > 0 else 0
            print(f"[{completed}/{len(tasks)}] {market} {month_str}: {len(records)} recs. Elapsed: {elapsed/60:.1f}m, Est rem: {rem/60:.1f}m")

    print(f"All MOEX data collection completed in {(time.time()-start_time)/60:.2f} minutes.")
    export_to_csv()

if __name__ == "__main__":
    main()
