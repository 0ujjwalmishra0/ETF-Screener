# 🧠 ETF Screener Pro 2026

A smart, automated **ETF Screener Dashboard** built in Python.
It fetches live ETF prices from **Yahoo Finance**, calculates key indicators (RSI, DMA, Z-Score), generates buy/sell signals, and produces a **beautiful color-coded dashboard** with sparkline charts — automatically opening every day at **8 AM IST**.

---

## 🚀 Features

✅ **Live Data Fetching**
Fetches latest ETF price data from Yahoo Finance (NSE-listed tickers).

✅ **Technical Indicators**

* **RSI (Relative Strength Index)** — measures momentum.
* **50DMA / 200DMA crossover** — detects long-term trend.
* **Z-Score** — shows overvaluation/undervaluation relative to mean.

✅ **Signal Engine**
Generates one of the following per ETF:
`STRONG BUY`, `BUY`, `WAIT`, `STRONG WAIT`
Includes **confidence %** and **trend direction** (📈 Up, ➡️ Side, 📉 Down).

✅ **Beautiful Output**

* Color-coded Excel file
* Rich HTML dashboard with:

  * Sparkline charts for each ETF
  * Confidence bars
  * Trend icons and Z-Score ranking
* Opens automatically when finished

✅ **Auto-Scheduler (Optional)**
Runs daily at **8 AM IST** to refresh data and rebuild dashboard.

---

## 📘 Technical Indicator Logic

### 🧮 1. **RSI (Relative Strength Index)**

RSI measures the speed and change of price movements.

**Formula:**

```
RSI = 100 - (100 / (1 + RS))
where RS = (Average Gain over 14 days) / (Average Loss over 14 days)
```

**Interpretation:**

| RSI Value | Meaning                 |
| --------- | ----------------------- |
| < 30      | Oversold (Bullish)      |
| 30–50     | Neutral to Mild Bullish |
| 50–70     | Caution Zone            |
| > 70      | Overbought (Bearish)    |

In this screener:

* RSI < 40 → Strong bullish momentum (+2 points)
* RSI < 50 → Moderate bullish (+1 point)

---

### 📈 2. **50DMA / 200DMA**

These are moving averages that smooth out short-term volatility.

**Logic:**

* **50DMA > 200DMA** → Bullish trend (+1 point)
* **Price > 50DMA × 1.01** → Strong short-term momentum (+1 point)

**Interpretation:**

| Pattern                                   | Signal  |
| ----------------------------------------- | ------- |
| Golden Cross (50DMA crosses above 200DMA) | Bullish |
| Death Cross (50DMA crosses below 200DMA)  | Bearish |

---

### ⚖️ 3. **Z-Score**

Z-Score measures how far current price is from its average (in standard deviations).

**Formula:**

```
Z = (Current Price - 50DMA) / RollingStandardDeviation(50 days)
```

**Interpretation:**

| Z-Score | Meaning                    |
| ------- | -------------------------- |
| < -1    | Undervalued (possible Buy) |
| -1 to 1 | Fairly valued (Hold/Wait)  |
| > 1     | Overvalued (possible Sell) |

This screener sorts ETFs by **Z-Score (ascending)** — lowest values appear first, highlighting potentially undervalued opportunities.

---

### 🧮 Scoring Logic Summary

| Condition            | Points |
| -------------------- | ------ |
| RSI < 40             | +2     |
| RSI < 50             | +1     |
| 50DMA > 200DMA       | +1     |
| Price > 50DMA × 1.01 | +1     |

Total max = 4 points → converted to **confidence %**

```
confidence = (score / 4) * 100
```

**Signal Mapping:**

| Score | Confidence | Signal      |
| ----- | ---------- | ----------- |
| 4     | 100%       | STRONG BUY  |
| 3     | 75%        | BUY         |
| 2     | 50%        | WAIT        |
| ≤1    | ≤25%       | STRONG WAIT |

---

## 💻 Example Output

| ETF           | Trend   | Sparkline           | Close   | RSI  | 50DMA | 200DMA | ZScore | Signal      | Confidence      |
| ------------- | ------- | ------------------- | ------- | ---- | ----- | ------ | ------ | ----------- | --------------- |
| NIFTYBEES.NS  | 📈 Up   | *(7-day sparkline)* | ₹291.04 | 43.2 | 280.1 | 270.3  | -1.23  | BUY         | ▓▓▓▓▓▓▓▓▓▓▓ 75% |
| SILVERBEES.NS | 📉 Down | *(7-day sparkline)* | ₹138.75 | 50.3 | 129.9 | 104.9  | +1.45  | STRONG WAIT | ▓▓▓▓ 25%        |

---

## 🧩 Project Structure

```
etf_screener/
├── main.py                # Entry point
├── scheduler_job.py       # Auto-refresh daily
├── data_fetch.py          # Fetch data via yfinance
├── indicators.py          # RSI, DMA, ZScore calculations
├── signals.py             # Signal + trend engine
├── sparkline_generator.py # Generates sparkline PNGs
├── output_writer.py       # Creates Excel + HTML dashboard
└── README.md              # Full documentation
```

---

## 📊 Output Files

| File                         | Description                         |
| ---------------------------- | ----------------------------------- |
| `ETF_Screener_Pro_2026.xlsx` | Excel with color-coded signal cells |
| `ETF_Screener_Pro_2026.html` | Interactive HTML dashboard          |
| `sparklines/*.png`           | Mini sparkline graphs for each ETF  |

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install yfinance matplotlib openpyxl pandas schedule
```

### 2️⃣ Run the Screener

```bash
python3 main.py
```

### 3️⃣ Optional — Auto Refresh (8AM IST)

```bash
python3 scheduler_job.py
```

---

## 💹 Default ETF Universe

| Category      | Example Tickers                                         |
| ------------- | ------------------------------------------------------- |
| Indian Equity | NIFTYBEES.NS, JUNIORBEES.NS, MID150BEES.NS, BANKBEES.NS |
| Commodities   | GOLDBEES.NS, SILVERBEES.NS                              |

You can modify the `TICKERS` list in `signals.py` to include:

* **Global ETFs** (e.g., `MON100.NS`, `MAFANG.NS`, `HNGSNGBEES.NS`)
* **Sector ETFs** (e.g., `PSUBANKBEES.NS`, `ITBEES.NS`, `PHARMABEES.NS`)

---

## 🧠 Example Signal Output in Console

```
🚀 Running ETF Screener Pro 2026...
     BUY 📈 Up → NIFTYBEES.NS: ₹291.04 | RSI=43.2 | Conf=75%
 STRONG WAIT 📉 Down → SILVERBEES.NS: ₹138.75 | RSI=58.1 | Conf=25%
✅ Screener run complete. Sorted by Z-Score.
🌐 Auto-opening dashboard...
```

---

## 📈 How to Interpret Z-Score in Dashboard

| Z-Score Range | ETF Status           | Action                         |
| ------------- | -------------------- | ------------------------------ |
| < -1          | Undervalued          | Consider buying / SIP addition |
| -1 to 0       | Slightly undervalued | Neutral                        |
| 0 to +1       | Slightly overvalued  | Caution                        |
| > +1          | Overvalued           | Avoid fresh buying             |

---

## 📦 requirements.txt

(optional file to create in project root)

```
yfinance==0.2.43
pandas==2.2.3
matplotlib==3.9.2
openpyxl==3.1.5
schedule==1.2.1
```

---

## 💡 Future Enhancements

* 🧾 Summary header (“3 Buys • Avg RSI = 47.3”)
* 📊 Trendline comparison across ETFs
* ☁️ Google Sheets live sync
* 🔔 Telegram/Email alerts when signal changes

---

## 🧑‍💻 Author

**Ujjwal Mishra**
Software Engineer — Java | Spring Boot | Python | Data Analytics
📧 0ujjwalmishra0@gmail.com
