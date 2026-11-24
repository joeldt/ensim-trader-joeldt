import csv, json, os, time
import numpy as np
import requests

MARKET_URL = "https://raw.githubusercontent.com/exofri/ensim-trade-center/refs/heads/main/docs/data/market.json"
HOLDINGS_URL = "https://raw.githubusercontent.com/exofri/ensim-trade-center/refs/heads/main/traders/joel/holdings.json"
LEADERBOARD_URL = "https://raw.githubusercontent.com/exofri/ensim-trade-center/refs/heads/main/docs/data/leaderboard.json"
MY_ID = "joel"
BASELINES = ["random_trader", "random_buyer", "random_seller"]
LOG_PATH = "docs/data/decision_log.csv"
PERF_PATH = "docs/data/performance_log.csv"
SIGNAL_PATH = "docs/data/signal.json"

MIN_MONEY_RESERVE = 50   # never let a buy signal spend below this much money
MIN_HISTORY = 40         # below this, not enough ticks to see a cycle: hold
WINDOW = 200             # only the most recent ticks are used for the fit
PERIODS = range(8, 200)  # cycle lengths tested, in ticks


def fetch_json(url):
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=(10, 20))
    resp.raise_for_status()
    return resp.json()


def fit_cycle(prices):
    """Fits price = level + trend * t + wave(period), for every period in PERIODS,
    and keeps the period with the smallest error. Returns (period, coefficients)."""
    y = np.array(prices)
    t = np.arange(len(y))
    best = None
    for period in PERIODS:
        if period > len(y):
            break
        X = np.column_stack([np.ones(len(y)), t,
                             np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        error = np.mean((X @ coef - y) ** 2)
        if best is None or error < best[0]:
            best = (error, period, coef)
    return best[1], best[2]


def predict(coef, period, t):
    return (coef[0] + coef[1] * t
            + coef[2] * np.sin(2 * np.pi * t / period)
            + coef[3] * np.cos(2 * np.pi * t / period))


def analyse(prices):
    """Returns (signal, details). Signal in [-1, 1]: > 0 buy, < 0 sell, 0 hold."""
    if len(prices) < MIN_HISTORY:
        return 0.0, {"reason": f"historique trop court ({len(prices)} ticks)"}
    recent = prices[-WINDOW:]
    period, coef = fit_cycle(recent)
    amplitude = float(np.hypot(coef[2], coef[3]))
    n = len(recent)                      # index of the next tick (where my order executes)
    horizon = max(1, int(period / 4))    # look a quarter of a cycle ahead
    price_next = predict(coef, period, n)
    price_later = np.mean([predict(coef, period, n + k) for k in range(1, horizon + 1)])
    expected_move = float(price_later - price_next)
    signal = float(np.clip(expected_move / amplitude, -1, 1))
    reason = ("hausse attendue, achat" if signal > 0.05 else
              "baisse attendue, vente" if signal < -0.05 else "pas de mouvement net attendu")
    # Fitted wave on the last ticks + forecast, saved for the dashboard
    shown = min(n, 120)
    fitted = [round(float(predict(coef, period, i)), 4) for i in range(n - shown, n)]
    forecast = [round(float(predict(coef, period, n + k)), 4) for k in range(0, horizon + 1)]
    return signal, {"period": int(period), "amplitude": round(amplitude, 3),
                    "expected_move": round(expected_move, 3), "reason": reason,
                    "fitted": fitted, "forecast": forecast}


def compute_signal(prices):
    return analyse(prices)[0]


def apply_budget_guard(raw_signal, money, price, reserve=MIN_MONEY_RESERVE):
    # Provided by the manual: a buy that would spend into the reserve is scaled down.
    if raw_signal <= 0:
        return raw_signal
    if money <= reserve:
        return 0.0
    max_affordable_units = (money - reserve) / price
    desired_units = 100 * raw_signal
    if desired_units <= max_affordable_units:
        return raw_signal
    return round(max_affordable_units / 100, 3)


def append_log(tick, decision, details, holdings):
    # One row per product and per market tick (the workflow runs more often than the market).
    already = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            already = {(r["tick"], r["product"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tick", "updated_at", "product", "signal", "period", "expected_move", "money", "units_held", "reason"])
        for p, d in details.items():
            if (str(tick), p) in already:
                continue
            w.writerow([tick, decision["updated_at"], p, decision[p], d.get("period", ""),
                        d.get("expected_move", ""), round(holdings["money"], 2),
                        round(holdings["products"][p], 2), d["reason"]])


def already_logged(path, tick):
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return any(r["tick"] == str(tick) for r in csv.DictReader(f))


def log_performance(tick, leaderboard):
    # One row per market tick: my total value and the three baselines' values.
    if already_logged(PERF_PATH, tick):
        return
    values = {r["id"]: r["total_value"] for r in leaderboard["leaderboard"]}
    ranking = [r["id"] for r in leaderboard["leaderboard"]]
    new_file = not os.path.exists(PERF_PATH)
    with open(PERF_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tick", "generated_at", "me"] + BASELINES + ["rank", "n_traders"])
        w.writerow([tick, leaderboard["generated_at"], values.get(MY_ID)]
                   + [values.get(b) for b in BASELINES]
                   + [ranking.index(MY_ID) + 1 if MY_ID in ranking else "", len(ranking)])


def main():
    market = fetch_json(MARKET_URL)
    holdings = fetch_json(HOLDINGS_URL)
    products = market["products"]
    history = market["history"]
    tick = history[-1]["t"] if history else 0
    os.makedirs("docs/data", exist_ok=True)

    decision, details = {}, {}
    money_left = holdings["money"]   # the market executes A, then B, then C with the same money
    for p in products:
        prices = [row[p] for row in history]
        raw, info = analyse(prices)
        current_price = prices[-1] if prices else 0
        decision[p] = round(apply_budget_guard(raw, money_left, current_price), 3)
        if decision[p] > 0:
            money_left -= 100 * decision[p] * current_price
        details[p] = info
        print(f"{p}: signal {decision[p]:+.3f}  period {info.get('period')}  {info['reason']}")

    decision["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open("decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    append_log(tick, decision, details, holdings)
    log_performance(tick, fetch_json(LEADERBOARD_URL))
    signal_data = {"tick": tick, "updated_at": decision["updated_at"], "money": holdings["money"],
                   "holdings": holdings["products"], "products": {}}
    for p in products:
        prices = [row[p] for row in history]
        shown = len(details[p].get("fitted", []))
        signal_data["products"][p] = {
            "signal": decision[p], **{k: v for k, v in details[p].items()},
            "ticks": [row["t"] for row in history[-shown:]] if shown else [],
            "prices": prices[-shown:] if shown else [],
        }
    with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(signal_data, f, indent=1)
    print(f"Tick {tick} -- wrote decision.json: {decision}")


if __name__ == "__main__":
    main()
