import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = r"C:\Users\user\Documents\Code\Lua\Work_script_3.0\Ur_lico\jelus_test"
LOG_FILE = os.path.join(FILE_PATH, "market_data.csv")

# Если файл в виндовс пути не найден, проверяем локальный лог
if not os.path.exists(LOG_FILE):
    LOG_FILE = "market_data.csv"

def generate_sample_log_if_missing():
    if not os.path.exists(LOG_FILE):
        data = """datetime,sec_code,last_price,bid,ask,implied_vol,realized_vol,vrp_spread,cbr_rate,advice
2026-09-08 12:00:00,SiM4,89500.00,89490.00,89510.00,18.50,14.20,4.30,18.00,ПРОДАВАТЬ VOL / SELL STRADDLE
2026-09-08 12:00:00,MXM4,3150.00,3149.00,3151.00,21.20,15.80,5.40,18.00,ПРОДАВАТЬ VOL / SELL STRADDLE
2026-09-08 12:00:00,RIM4,1120.00,1119.50,1120.50,22.80,18.10,4.70,18.00,ПРОДАВАТЬ VOL / SELL STRADDLE
2026-09-08 12:00:00,GDM4,2510.00,2509.50,2510.50,19.40,16.20,3.20,18.00,ПРОДАВАТЬ VOL / SELL STRADDLE
2026-09-08 12:00:00,SRM4,28500.00,28490.00,28510.00,24.50,23.80,0.70,18.00,HOLD / НЕЙТРАЛЬНО
"""
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(data)

def process_and_visualize():
    generate_sample_log_if_missing()

    print("=================================================================")
    print(" QUIK Advisory & Visualization Engine (Jelus Test) ")
    print("=================================================================")
    print(f"Чтение файла логов: {LOG_FILE}\n")

    df = pd.read_csv(LOG_FILE)
    latest_df = df.groupby('sec_code').last().reset_index()

    print("---------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Инструмент':<10} | {'Цена Last':<10} | {'IV (%)':<8} | {'RV (%)':<8} | {'VRP (%)':<8} | {'Ставка ЦБ':<10} | {'ТОРГОВЫЙ СОВЕТ / СТРАТЕГИЯ'}")
    print("---------------------------------------------------------------------------------------------------------------------------------")

    for _, row in latest_df.iterrows():
        sec = row['sec_code']
        price = row['last_price']
        iv = row['implied_vol']
        rv = row['realized_vol']
        vrp = row['vrp_spread']
        cbr = row['cbr_rate']
        advice = row['advice']

        # Расчет конкретных цен входа, стоп-лосса и тейк-профита
        entry_price = price
        stop_loss = price * 0.985 if "BUY" in advice else price * 1.015
        take_profit = price * 1.03 if "BUY" in advice else price * 0.97

        print(f"{sec:<10} | {price:<10.2f} | {iv:<8.2f} | {rv:<8.2f} | {vrp:<8.2f} | {cbr:<10.2f} | {advice}")
        print(f"  --> ДЕТАЛИ СДЕЛКИ: Вход: {entry_price:.2f} РУБ | Stop-Loss: {stop_loss:.2f} РУБ | Take-Profit: {take_profit:.2f} РУБ")
        print("---------------------------------------------------------------------------------------------------------------------------------")

    # Визуализация
    plt.figure(figsize=(12, 6))

    x = np.arange(len(latest_df))
    width = 0.35

    plt.bar(x - width/2, latest_df['implied_vol'], width, label='Implied Volatility (IV % - Опционы)', color='crimson')
    plt.bar(x + width/2, latest_df['realized_vol'], width, label='Realized Volatility (RV % - Фьючерс)', color='navy')

    plt.xlabel('Фьючерсные Инструменты MOEX', fontsize=11)
    plt.ylabel('Волатильность (%)', fontsize=11)
    plt.title('Сравнение Подразумеваемой (IV) и Реализованной (RV) Волатильности из QUIK', fontsize=13)
    plt.xticks(x, latest_df['sec_code'])
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("quik_volatility_chart.png", dpi=300)
    plt.close()
    print("\nГрафик волатильности сохранен в quik_volatility_chart.png")

if __name__ == "__main__":
    process_and_visualize()
