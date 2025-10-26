import schedule, time
from datetime import datetime
from signals import run_once

def job():
    print(f"\n⏰ Running ETF Screener at {datetime.now()}...\n")
    run_once()

def schedule_daily_job():
    # 8 AM IST = 02:30 UTC
    schedule.every().day.at("02:30").do(job)
    print("🕗 Auto-refresh scheduled at 8 AM IST daily.")
    job()  # Run immediately once
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    schedule_daily_job()
