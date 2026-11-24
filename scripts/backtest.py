"""Replays the real price history to compare my cycle signal with simpler strategies.
Each decision only uses the prices known before the tick where the order executes.
Simplification: my own orders are assumed not to move the price."""
import io, json, random
import numpy as np
import requests
from algorithm import compute_signal, MIN_MONEY_RESERVE

HISTORY_CSV = "https://raw.githubusercontent.com/exofri/ensim-trade-center/refs/heads/main/market/price_history.csv"
START = 150          # first simulated tick (earlier ticks are the model's warm-up)
N_RANDOM_RUNS = 30   # random baselines are averaged over several draws


def load_prices():
    text = requests.get(HISTORY_CSV, timeout=(10, 30)).text
    rows = [line.split(",") for line in text.strip().split("\n")[1:]]
    ticks = [int(r[0]) for r in rows]
    prices = {p: np.array([float(r[2 + i]) for r in rows]) for i, p in enumerate("ABC")}
    return ticks, prices


def bollinger(y, window=20):
    if len(y) < window:
        return 0.0
    z = (y[-1] - np.mean(y[-window:])) / (np.std(y[-window:]) + 1e-9)
    return float(np.clip(-z / 2, -1, 1))   # far above the mean -> sell, far below -> buy


def momentum(y, window=5):
    if len(y) < window + 1:
        return 0.0
    return float(np.clip((y[-1] - y[-1 - window]) / y[-1 - window] * 10, -1, 1))


def simulate(prices, decide, rng=None):
    money, units = 1000.0, {p: 1000.0 for p in "ABC"}   # starting capital of a new trader
    n = len(prices["A"])
    values = []
    for t in range(START, n):
        for p in "ABC":
            price = prices[p][t]
            d = decide(p, prices[p][:t], rng)
            if d > 0:   # same budget guard as algorithm.py
                d = 0.0 if money <= MIN_MONEY_RESERVE else min(d, (money - MIN_MONEY_RESERVE) / price / 100)
                bought = min(100 * d, money / price)
                money -= bought * price
                units[p] += bought
            elif d < 0:
                sold = min(100 * -d, units[p])
                money += sold * price
                units[p] -= sold
        values.append(money + sum(units[p] * prices[p][t] for p in "ABC"))
    return np.array(values)


def summary(values):
    drawdown = values / np.maximum.accumulate(values) - 1
    returns = np.diff(values) / values[:-1]
    return {"return_pct": round(float(values[-1] / values[0] - 1) * 100, 1),
            "max_drawdown_pct": round(float(drawdown.min()) * 100, 1),
            "volatility_pct": round(float(np.std(returns)) * 100, 2)}


def main():
    ticks, prices = load_prices()
    strategies = {
        "Modèle de cycle (mon signal)": lambda p, y, r: compute_signal(list(y)),
        "Bandes de Bollinger": lambda p, y, r: bollinger(y),
        "Momentum": lambda p, y, r: momentum(y),
        "Ne rien faire": lambda p, y, r: 0.0,
    }
    randoms = {
        "random_trader": lambda p, y, r: r.uniform(-1, 1),
        "random_buyer": lambda p, y, r: r.uniform(0, 1),
        "random_seller": lambda p, y, r: r.uniform(-1, 0),
    }
    out = {"ticks": ticks[START:], "curves": {}, "summary": {}}
    for name, f in strategies.items():
        v = simulate(prices, f)
        out["curves"][name] = [round(x, 1) for x in v]
        out["summary"][name] = summary(v)
        print(name, out["summary"][name])
    for name, f in randoms.items():
        runs = [simulate(prices, f, random.Random(seed)) for seed in range(N_RANDOM_RUNS)]
        v = np.mean(runs, axis=0)
        out["curves"][name] = [round(x, 1) for x in v]
        out["summary"][name] = summary(v)
        print(name, "(average)", out["summary"][name])
    with open("docs/data/backtest.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("Wrote docs/data/backtest.json")


if __name__ == "__main__":
    main()
