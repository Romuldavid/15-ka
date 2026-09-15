import csv
import os

def verify_all():
    print("=== Running Final Verification ===")

    # 1. CBR
    assert os.path.exists("cbr_key_rate.csv"), "cbr_key_rate.csv missing"
    with open("cbr_key_rate.csv", "r", encoding="utf-8") as f:
        cbr_rows = list(csv.DictReader(f))
    print(f"CBR key rate records: {len(cbr_rows)} (Range: {cbr_rows[0]['date']} to {cbr_rows[-1]['date']})")
    assert len(cbr_rows) > 0, "CBR data empty"

    # 2. Futures OHLC
    assert os.path.exists("moex_futures_ohlc.csv"), "moex_futures_ohlc.csv missing"
    with open("moex_futures_ohlc.csv", "r", encoding="utf-8") as f:
        fut_rows = list(csv.DictReader(f))
    fut_dates = [r['tradedate'] for r in fut_rows]
    print(f"Futures OHLC records: {len(fut_rows)} (Range: {min(fut_dates)} to {max(fut_dates)})")
    assert len(fut_rows) > 0, "Futures data empty"

    # 3. Options OHLC
    assert os.path.exists("moex_options_ohlc.csv"), "moex_options_ohlc.csv missing"
    with open("moex_options_ohlc.csv", "r", encoding="utf-8") as f:
        opt_rows = list(csv.DictReader(f))
    opt_dates = [r['tradedate'] for r in opt_rows]
    print(f"Options OHLC records: {len(opt_rows)} (Range: {min(opt_dates)} to {max(opt_dates)})")
    assert len(opt_rows) > 0, "Options data empty"

    # 4. Report and Chart
    assert os.path.exists("strategy_report.md"), "strategy_report.md missing"
    assert os.path.exists("pnl_chart.png"), "pnl_chart.png missing"
    print("Report and PnL Chart exist.")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_all()
