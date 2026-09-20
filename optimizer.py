"""숭이 전략 파라미터 그리드 서치 (병렬 연산 + SQLite 캐싱)."""
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import backtest
import strategies
from data_loader import fetch_ohlcv
from database import init_db, load_optimization, save_optimization

STRATEGY = "SoongI_3"
TICKERS = ["BTC-USD", "NVDA", "AAPL", "QQQ", "SPY", "SLV", "SOXX"]

# ---- 기본 모드: 익절/물타기 4개 파라미터 ----
GRID = {
    "tp1": [5, 10, 15, 20, 25, 30],            # 1차 반익절
    "tp2": [30, 40, 50, 60, 70, 80, 90, 100],  # 2차 완익절
    "drop_rate": [5, 10, 15, 20],              # 물타기 간격
    "buy_ratio": [10, 20, 25, 50],             # 1회 추가매수 수량 비율
}

# ---- 세분화 모드: 진입·물타기·청산 6개 파라미터 ----
# sma_period 는 '종가 <= SMA(n)' 를 RSI 조건과 OR 로 묶는 보조 진입 경로다.
# (숭이 3호 기본값은 SMA 미사용이므로 이 모드에는 기본값 조합이 포함되지 않는다)
DETAIL_GRID = {
    "rsi_buy": [45, 50, 55, 60],
    "sma_period": [10, 20, 30],
    "buy_ratio": [20, 30, 40, 50],
    "drop_rate": [5, 8, 10, 12],
    "tp1": [8, 10, 12, 15],
    "trailing_stop": [8, 10, 12, 15],
}
MODES = {
    "basic": {"label": "기본 (익절·물타기)", "grid": GRID,
              "cols": ["tp1", "tp2", "drop_rate", "buy_ratio"]},
    "detail": {"label": "숭이 3호 세부 최적화", "grid": DETAIL_GRID,
               "cols": ["rsi_buy", "sma_period", "buy_ratio", "drop_rate",
                        "tp1", "trailing_stop"]},
}
PARAM_KEYS = ["tp1", "tp2", "drop_rate", "buy_ratio",
              "rsi_buy", "sma_period", "trailing_stop"]

SL_LEVELS = list(strategies.STOP_LOSS_LEVELS) + [15.0]

END = date.today()
PERIODS = {
    "6M": END - timedelta(days=182),
    "1Y": END - timedelta(days=365),
    "3Y": END - timedelta(days=365 * 3),
    "5Y": END - timedelta(days=365 * 5),
}
FETCH_START = date(2019, 10, 1)


def combos():
    """기본 모드. tp1 >= tp2 인 불합리 조합은 건너뛴다."""
    out = []
    for tp1 in GRID["tp1"]:
        for tp2 in GRID["tp2"]:
            if tp1 >= tp2:
                continue
            for dr in GRID["drop_rate"]:
                for br in GRID["buy_ratio"]:
                    out.append({"tp1": tp1, "tp2": tp2,
                                "drop_rate": dr, "buy_ratio": br})
    return out


def detail_combos():
    """세분화 모드 — 6개 파라미터 전수 조합."""
    g = DETAIL_GRID
    return [{"rsi_buy": r, "sma_period": s, "buy_ratio": b, "drop_rate": d,
             "tp1": t, "trailing_stop": tr}
            for r in g["rsi_buy"] for s in g["sma_period"] for b in g["buy_ratio"]
            for d in g["drop_rate"] for t in g["tp1"] for tr in g["trailing_stop"]]


def mode_combos(mode):
    return combos() if mode == "basic" else detail_combos()


TOTAL = len(combos())
TOTAL_DETAIL = len(detail_combos())
MODES["basic"]["total"] = TOTAL
MODES["detail"]["total"] = TOTAL_DETAIL

# ---- 워커 프로세스 전역 (initializer 로 1회만 로드) ----
_PRICES = {}


def _init_worker(tickers, start, end):
    global _PRICES
    _PRICES = {t: fetch_ohlcv(t, start, end) for t in tickers}


def _make(p, sl):
    """조합 딕셔너리를 숭이 3호 인스턴스에 주입한다.
    지정되지 않은 파라미터는 클래스 기본값을 그대로 쓴다."""
    s = strategies.SoongI3()
    if "tp1" in p:
        s.TP1 = p["tp1"] / 100
    if "tp2" in p:
        s.TP2 = p["tp2"] / 100
    if "drop_rate" in p:
        s.DROP_STEP = p["drop_rate"] / 100
    if "buy_ratio" in p:
        s.ADD_RATIO = p["buy_ratio"] / 100
    if "rsi_buy" in p:
        s.RSI_BUY = p["rsi_buy"]
    if "sma_period" in p:
        s.SMA_PERIOD = int(p["sma_period"])
    if "trailing_stop" in p:
        s.TRAIL = p["trailing_stop"] / 100
    s.SL = -abs(sl) / 100
    return s


def _eval(args):
    """한 조합을 대상 종목 전체에 대해 실행하고 평균 지표를 낸다."""
    params, tickers, start, end, sl = args
    strat = _make(params, sl)
    acc, n = {}, 0
    for t in tickers:
        r, _, _ = backtest.run(_PRICES[t], t, STRATEGY, start=start, end=end,
                               stop_loss_pct=sl, strat=strat)
        n += 1
        for k in ("total_return_pct", "mdd_pct", "sharpe_ratio", "sortino_ratio",
                  "cagr_pct", "calmar_ratio", "win_rate_pct", "total_trades"):
            acc[k] = acc.get(k, 0) + r[k]
    out = {k: (v if k == "total_trades" else round(v / n, 3)) for k, v in acc.items()}
    out.update({k: params.get(k, 0) for k in PARAM_KEYS})
    return out


def run_grid(scope="ALL", period="3Y", sl=15.0, force=False, workers=None,
             mode="basic"):
    """그리드 서치 실행. 캐시가 완전하면 즉시 반환."""
    if period not in PERIODS:
        raise ValueError(f"지원하지 않는 기간: {period}")
    if mode not in MODES:
        raise ValueError(f"지원하지 않는 모드: {mode}")
    init_db()
    total = MODES[mode]["total"]
    cached = load_optimization(STRATEGY, scope, period, sl, mode)
    if not force and len(cached) >= total:
        return cached, True

    tickers = TICKERS if scope == "ALL" else [scope]
    start, end = PERIODS[period].isoformat(), END.isoformat()
    args = [(c, tickers, start, end, sl) for c in mode_combos(mode)]
    # 무료 클라우드는 CPU·메모리가 작다. OPT_WORKERS 로 상한을 둘 수 있다.
    cap = int(os.getenv("OPT_WORKERS", "16"))
    workers = workers or max(1, min(cap, (os.cpu_count() or 4) - 1))

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(tickers, FETCH_START.isoformat(), end)) as ex:
        results = list(ex.map(_eval, args, chunksize=8))

    rows = [dict(r, strategy_name=STRATEGY, mode=mode, scope=scope, period=period,
                 stop_loss_pct=sl) for r in results]
    save_optimization(rows)
    return rows, False


LABELS = {"tp1": "1차익절", "tp2": "2차익절", "drop_rate": "물타기",
          "buy_ratio": "매수량", "rsi_buy": "RSI", "sma_period": "SMA",
          "trailing_stop": "트레일링"}


def summarize(rows, mode="basic", title=""):
    """Calmar / 수익률 / MDD 1위 조합을 표로 출력."""
    cols = MODES[mode]["cols"]
    picks = [("① 최고 Calmar", max(rows, key=lambda r: r["calmar_ratio"])),
             ("② 최고 수익률", max(rows, key=lambda r: r["total_return_pct"])),
             ("③ 최저 MDD", max(rows, key=lambda r: r["mdd_pct"]))]
    if title:
        print(f"\n{title}\n" + "=" * 104)
    print(f"{'구분':<15}" + "".join(f"{LABELS[c]:>9}" for c in cols)
          + f"{'수익률%':>11}{'MDD%':>9}{'Sharpe':>9}{'Calmar':>8}{'승률%':>8}")
    print("-" * 104)
    for lab, r in picks:
        print(f"{lab:<15}" + "".join(f"{r[c]:>9.0f}" for c in cols)
              + f"{r['total_return_pct']:>11.2f}{r['mdd_pct']:>9.2f}"
              f"{r['sharpe_ratio']:>9.3f}{r['calmar_ratio']:>8.2f}"
              f"{r['win_rate_pct']:>8.1f}")
    return picks


if __name__ == "__main__":
    import sys
    import time
    sc = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    pe = sys.argv[2] if len(sys.argv) > 2 else "3Y"
    s = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    md = sys.argv[4] if len(sys.argv) > 4 else "basic"
    t0 = time.perf_counter()
    rows, hit = run_grid(sc, pe, s, mode=md)
    summarize(rows, md, f"[{MODES[md]['label']}] {sc} / {pe} / 손절 {s:g}% — "
                        f"{len(rows)}조합, "
                        f"{'캐시' if hit else f'{time.perf_counter()-t0:.1f}초 연산'}")
