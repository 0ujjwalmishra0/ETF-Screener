from flask import Flask, render_template, request, redirect, url_for
from signals import run_once
from portfolio_tracker import PortfolioTracker
from output_writer import generate_dashboard_html
import json, os

app = Flask(__name__)
PORTFOLIO_FILE = "portfolio.json"

@app.route("/")
def home():
    df = run_once()
    html = generate_dashboard_html(df)
    return html

@app.route("/add_to_portfolio", methods=["POST"])
def add_to_portfolio():
    etf = request.form.get("etf")
    price = float(request.form.get("price"))

    portfolio = []
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            portfolio = json.load(f)

    portfolio.append({"ETF": etf, "BuyPrice": price})
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

    print(f"✅ Added {etf} @ ₹{price}")
    return redirect(url_for("home"))

@app.route("/portfolio")
def view_portfolio():
    tracker = PortfolioTracker()
    df = tracker.evaluate_positions()
    return render_template("portfolio.html", df=df)
