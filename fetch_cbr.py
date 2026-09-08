import urllib.request
import re
import csv
import json
from datetime import datetime

def fetch_cbr_key_rate(from_date="01.01.2023", to_date="08.09.2026"):
    url = f"https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From={from_date}&UniDbQuery.To={to_date}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8')

    rows = re.findall(r'<tr>\s*<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d\s,]+)</td>\s*</tr>', html)

    parsed_data = []
    for date_str, rate_str in rows:
        d_obj = datetime.strptime(date_str, '%d.%m.%Y')
        rate_val = float(rate_str.replace(' ', '').replace(',', '.'))
        parsed_data.append({
            'date': d_obj.strftime('%Y-%m-%d'),
            'date_ru': date_str,
            'key_rate': rate_val
        })

    # Sort chronologically
    parsed_data.sort(key=lambda x: x['date'])
    return parsed_data

def main():
    print("Fetching CBR Key Rate from 01.01.2023 to 08.09.2026...")
    data = fetch_cbr_key_rate()
    print(f"Retrieved {len(data)} entries. Earliest: {data[0]['date']} ({data[0]['key_rate']}%), Latest: {data[-1]['date']} ({data[-1]['key_rate']}%)")

    # Save CSV
    csv_file = "cbr_key_rate.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "date_ru", "key_rate"])
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {csv_file}")

    # Save JSON
    json_file = "cbr_key_rate.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {json_file}")

if __name__ == "__main__":
    main()
