import sqlite3
import math
import csv
import re
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 1e-5:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    if sigma <= 1e-5:
        sigma = 1e-5
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    if T <= 1e-5 or market_price <= 0:
        return 0.25
    low, high = 0.01, 4.0
    for _ in range(30):
        mid = (low + high) / 2.0
        price = bs_price(S, K, T, r, mid, option_type)
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

class SeparateRegimeAnalyzer:
    def __init__(self, db_path="moex_market_data.db", cbr_path="cbr_key_rate.csv", initial_capital=1000000.0):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.cbr_rates = self.load_cbr_rates(cbr_path)
        self.prefixes = ['RI', 'MX', 'Si', 'CR', 'BR', 'NG', 'GD', 'SV', 'SR', 'GZ', 'LK', 'VB', 'YN', 'GK', 'RN', 'NK', 'PZ']

    def load_cbr_rates(self, cbr_path):
        rates = {}
        with open(cbr_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rates[row['date']] = float(row['key_rate']) / 100.0
        return rates

    def get_risk_free_rate(self, date_str):
        if date_str in self.cbr_rates:
            return self.cbr_rates[date_str]
        past_dates = [d for d in self.cbr_rates.keys() if d <= date_str]
        return self.cbr_rates[max(past_dates)] if past_dates else 0.10

    def parse_secid(self, secid, prefix):
        m = re.match(rf'^{prefix}(\d+(?:\.\d+)?)[A-Z0-9]+$', secid, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except:
                return None
        return None

    def run_regimes(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        call_debit_results = []
        put_credit_results = []

        for prefix in self.prefixes:
            c.execute("SELECT tradedate, secid, close, volume, numtrades FROM options_ohlc WHERE secid LIKE ? ORDER BY tradedate", (f"{prefix}%",))
            opt_rows = c.fetchall()
            if not opt_rows:
                continue

            opts_by_date = {}
            for td, secid, cl, vol, nt in opt_rows:
                strike = self.parse_secid(secid, prefix)
                if strike is not None and nt > 0:
                    if td not in opts_by_date:
                        opts_by_date[td] = []
                    opts_by_date[td].append({'secid': secid, 'strike': strike, 'close': cl, 'volume': vol})

            c.execute("SELECT tradedate, secid, close FROM futures_ohlc WHERE secid LIKE ? ORDER BY tradedate", (f"{prefix}%",))
            fut_rows = c.fetchall()
            futs_by_date = {}
            for td, fsec, fcl in fut_rows:
                if td not in futs_by_date:
                    futs_by_date[td] = []
                futs_by_date[td].append({'secid': fsec, 'close': fcl})

            trading_dates = sorted(list(set(opts_by_date.keys()).intersection(set(futs_by_date.keys()))))
            if len(trading_dates) < 15:
                continue

            spot_prices = [futs_by_date[d][0]['close'] for d in trading_dates]
            spot_df = pd.DataFrame({'date': trading_dates, 'spot': spot_prices})
            spot_df['ema20'] = spot_df['spot'].ewm(span=20).mean()
            spot_df['trend'] = spot_df['spot'] > spot_df['ema20']

            iv_history = []
            for idx, d_str in enumerate(trading_dates):
                r_rate = self.get_risk_free_rate(d_str)
                spot = spot_prices[idx]
                opts = opts_by_date[d_str]
                closest = min(opts, key=lambda x: abs(x['strike'] - spot))
                iv = implied_volatility(closest['close'], spot, closest['strike'], 30/252.0, r_rate)
                iv_history.append(iv)

            spot_df['iv'] = iv_history
            spot_df['iv_rank'] = (spot_df['iv'] - spot_df['iv'].rolling(30, min_periods=5).min()) / (
                spot_df['iv'].rolling(30, min_periods=5).max() - spot_df['iv'].rolling(30, min_periods=5).min() + 1e-6
            )

            # 1. Bull Call Spread (Debit Spread, Low IV Rank <= 50%)
            call_trades, call_wins, call_pnl = 0, 0, 0.0
            # 2. Bull Put Spread (Credit Spread, High IV Rank > 50%)
            put_trades, put_wins, put_pnl = 0, 0, 0.0

            for idx in range(0, len(trading_dates) - 10, 5):
                d_str = trading_dates[idx]
                spot = spot_prices[idx]
                opts = opts_by_date[d_str]
                is_bullish = spot_df.loc[idx, 'trend']
                iv_rank = spot_df.loc[idx, 'iv_rank']

                if not is_bullish:
                    continue # Both require bullish trend

                strikes = sorted(list(set(o['strike'] for o in opts)))
                if len(strikes) < 2:
                    continue

                atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
                if atm_idx >= len(strikes) - 1:
                    continue

                K1 = strikes[atm_idx]
                K2 = strikes[atm_idx + 1]

                opt1 = next((o for o in opts if abs(o['strike'] - K1) < 1e-4), None)
                opt2 = next((o for o in opts if abs(o['strike'] - K2) < 1e-4), None)

                if opt1 and opt2:
                    exp_idx = min(idx + 10, len(trading_dates) - 1)
                    exp_spot = spot_prices[exp_idx]

                    # Low IV: Bull Call Debit Spread
                    if pd.isna(iv_rank) or iv_rank <= 0.50:
                        debit = opt1['close'] - opt2['close']
                        if 0 < debit < (K2 - K1):
                            payoff = max(0.0, exp_spot - K1) - max(0.0, exp_spot - K2)
                            pnl = payoff - debit
                            call_trades += 1
                            call_pnl += pnl
                            if pnl > 0:
                                call_wins += 1

                    # High IV: Bull Put Credit Spread
                    elif iv_rank > 0.50:
                        # Sell Put K2, Buy Put K1
                        # Estimate Put prices using Put-Call Parity if explicit puts not isolated
                        # Credit received = P_K2 - P_K1
                        credit = (opt2['close'] - spot + K2) - (opt1['close'] - spot + K1)
                        if credit > 0:
                            loss = max(0.0, K2 - exp_spot) - max(0.0, K1 - exp_spot)
                            pnl = credit - loss
                            put_trades += 1
                            put_pnl += pnl
                            if pnl > 0:
                                put_wins += 1

            call_debit_results.append({
                'prefix': prefix,
                'trades': call_trades,
                'win_rate': (call_wins / call_trades * 100) if call_trades > 0 else 0.0,
                'pnl': call_pnl
            })

            put_credit_results.append({
                'prefix': prefix,
                'trades': put_trades,
                'win_rate': (put_wins / put_trades * 100) if put_trades > 0 else 0.0,
                'pnl': put_pnl
            })

        conn.close()
        return call_debit_results, put_credit_results

def generate_individual_reports(call_results, put_results):
    df_call = pd.DataFrame(call_results)
    df_put = pd.DataFrame(put_results)

    # 1. Bull Call Debit Spread Report
    total_call_pnl = df_call['pnl'].sum()
    avg_call_win = df_call['win_rate'].mean()
    call_report = f"""# Отчет по стратегии «Bull Call Spread (Дебетовый бычий спрэд)»

## 1. Резюме стратегии
- **Режим рынка:** Низкая волатильность (**IV Rank <= 50%**) и бычий тренд (Цена > EMA-20)
- **Конструкция:** Покупка ATM Call (K1) + Продажа OTM Call (K2)
- **Характер позиции:** Чистый дебет (платится премия), чистая положительная дельта (Long Delta)
- **Суммарный PnL по MOEX:** `{total_call_pnl:,.2f} РУБ`
- **Средний Win Rate:** `{avg_call_win:.2f}%`

---

## 2. Результаты по Инструментам MOEX (01.01.2023 — 08.09.2026)

| Инструмент | Число сделок | Win Rate (%) | PnL (РУБ) |
|------------|--------------|--------------|-----------|
"""
    for _, r in df_call.iterrows():
        call_report += f"| {r['prefix']} | {r['trades']} | {r['win_rate']:.1f}% | {r['pnl']:,.2f} |\n"

    call_report += """
---

## 3. Выводы по Дебетовому Бычьему Спрэду
1. **Преимущества:** Ограниченный риск (не больше уплаченного дебета) и высокий Leverage при низких ценах на покупку опционов.
2. **Ключевой фактор успеха:** Входить только в фазах низкой волатильности, чтобы избежать падения стоимости опционов от Vega-распада при падении IV.
"""

    with open("bull_call_debit_report.md", "w", encoding="utf-8") as f:
        f.write(call_report)
    print("Saved bull_call_debit_report.md")

    # 2. Bull Put Credit Spread Report
    total_put_pnl = df_put['pnl'].sum()
    avg_put_win = df_put['win_rate'].mean()
    put_report = f"""# Отчет по стратегии «Bull Put Spread (Кредитный бычий спрэд)»

## 1. Резюме стратегии
- **Режим рынка:** Высокая волатильность (**IV Rank > 50%**) и бычий тренд (Цена > EMA-20)
- **Конструкция:** Продажа OTM Put (K2) + Покупка дальше OTM Put (K1)
- **Характер позиции:** Чистый кредит (получается премия), положительный Theta (заработок на распаде времени)
- **Суммарный PnL по MOEX:** `{total_put_pnl:,.2f} РУБ`
- **Средний Win Rate:** `{avg_put_win:.2f}%`

---

## 2. Результаты по Инструментам MOEX (01.01.2023 — 08.09.2026)

| Инструмент | Число сделок | Win Rate (%) | PnL (РУБ) |
|------------|--------------|--------------|-----------|
"""
    for _, r in df_put.iterrows():
        put_report += f"| {r['prefix']} | {r['trades']} | {r['win_rate']:.1f}% | {r['pnl']:,.2f} |\n"

    put_report += """
---

## 3. Выводы по Кредитному Бычьему Спрэду
1. **Преимущества:** Высокая вероятность успеха (Win Rate), прибыль зарабатывается за счет временного распада (Theta) и «сжатия» волатильности (IV Crush).
2. **Ключевой фактор успеха:** Продавать дорогую волатильность на пиках IV с обязательной защитой купленным страйком ниже.
"""

    with open("bull_put_credit_report.md", "w", encoding="utf-8") as f:
        f.write(put_report)
    print("Saved bull_put_credit_report.md")

def main():
    analyzer = SeparateRegimeAnalyzer()
    call_res, put_res = analyzer.run_regimes()
    generate_individual_reports(call_res, put_res)

if __name__ == "__main__":
    main()
