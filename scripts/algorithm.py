import csv, json, os, time
import numpy as np
import requests

MARKET_URL = "https://raw.githubusercontent.com/exofri/ensim-trade-center/refs/heads/main/docs/data/market.json"
HOLDINGS_URL = "https://raw.githubusercontent.com/exofri/ensim-trade-center/refs/heads/main/traders/joel/holdings.json"
LOG_PATH = "decision_log.csv"

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
    return signal, {"period": int(period), "amplitude": round(amplitude, 3),
                    "expected_move": round(expected_move, 3), "reason": reason}


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


def main():
    market = fetch_json(MARKET_URL)
    holdings = fetch_json(HOLDINGS_URL)
    products = market["products"]
    history = market["history"]
    tick = history[-1]["t"] if history else 0

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
        print(f"{p}: signal {decision[p]:+.3f}  {info}")

    decision["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open("decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    append_log(tick, decision, details, holdings)
    print(f"Tick {tick} -- wrote decision.json: {decision}")


if __name__ == "__main__":
    main()
