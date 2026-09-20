"""FastAPI + Chart.js 백테스트 대시보드 (http://localhost:8000).

APScheduler 로 매일 한국시간 07:00(미 증시 마감) / 09:05(BTC 마감)에
백테스트를 자동 재실행한다. ENABLE_SCHEDULER=0 으로 끌 수 있다.
"""
import json
import os
import threading
import time
from pathlib import Path

from datetime import date, timedelta

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException

import backtest
from fastapi.responses import FileResponse

import optimizer
import run_backtest
import strategies
from data_loader import fetch_ohlcv
from database import all_meta, get_conn, has_results, init_db, set_meta

app = FastAPI(title="Backtest Dashboard")
BASE = Path(__file__).parent

STRATEGY_LABELS = strategies.LABELS
STOP_LOSS_LEVELS = list(strategies.STOP_LOSS_LEVELS)
DEFAULT_SL = strategies.DEFAULT_STOP_LOSS_PCT


# ---------------------------------------------------------------- 스케줄러
# 한국시간 기준 07:00 (미 증시 마감 정산) / 09:05 (BTC 일봉 마감 정산)
SCHEDULE = [("us_close", 7, 0), ("btc_close", 9, 5)]
TZ = os.getenv("SCHEDULER_TZ", "Asia/Seoul")
SCHEDULER_ENABLED = os.getenv("ENABLE_SCHEDULER", "1") != "0"
scheduler: BackgroundScheduler | None = None
_job_lock = threading.Lock()


def _run_backtests(trigger: str):
    """스케줄러/부트스트랩 공용 실행기. 중복 실행을 막는다."""
    if not _job_lock.acquire(blocking=False):
        print(f"[scheduler] {trigger}: 이미 실행 중 — 건너뜀", flush=True)
        return
    try:
        print(f"[scheduler] {trigger}: 백테스트 시작", flush=True)
        r = run_backtest.run_all_backtests(report_output=False, trigger=trigger)
        print(f"[scheduler] {trigger}: 완료 {r['rows']}건 / {r['seconds']}초 "
              f"(데이터 {r['last_price_date']})", flush=True)
    except Exception as exc:                       # 실패해도 서버는 계속 뜬다
        set_meta("last_backtest_error", f"{trigger}: {exc}")
        print(f"[scheduler] {trigger}: 실패 — {exc}", flush=True)
    finally:
        _job_lock.release()


@app.on_event("startup")
def _startup():
    global scheduler
    init_db()

    # 배포 직후(빈 DB)에는 백그라운드로 1회 채운다 — 기동을 막지 않는다
    if not has_results():
        print("[startup] 백테스트 결과 없음 — 부트스트랩 실행", flush=True)
        threading.Thread(target=_run_backtests, args=("bootstrap",),
                         daemon=True).start()

    if not SCHEDULER_ENABLED:
        print("[startup] ENABLE_SCHEDULER=0 — 스케줄러 비활성화", flush=True)
        return
    scheduler = BackgroundScheduler(timezone=TZ)
    for name, hh, mm in SCHEDULE:
        scheduler.add_job(_run_backtests, CronTrigger(hour=hh, minute=mm, timezone=TZ),
                          args=[name], id=name, replace_existing=True,
                          max_instances=1, misfire_grace_time=3600,
                          coalesce=True)
    scheduler.start()
    for j in scheduler.get_jobs():
        print(f"[scheduler] 등록: {j.id} — 다음 실행 {j.next_run_time}", flush=True)


@app.on_event("shutdown")
def _shutdown():
    if scheduler:
        scheduler.shutdown(wait=False)


PERIOD_META = {
    "6M":   {"label": "최근 6개월", "note": "표본이 매우 짧아 참고용"},
    "1Y":   {"label": "최근 1년", "note": "표본이 짧아 순위는 노이즈"},
    "3Y":   {"label": "최근 3년", "note": "2023-09~ · 상승장"},
    "5.7Y": {"label": "2021-01~현재", "note": "2022 하락장 포함"},
}
DEFAULT_PERIOD = "3Y"


def _periods() -> list:
    with get_conn() as conn:
        found = [r["period"] for r in conn.execute(
            "SELECT DISTINCT period FROM backtest_results")]
    order = list(PERIOD_META)
    return sorted(found, key=lambda p: order.index(p) if p in order else 99)


def _period(value: str | None) -> str:
    avail = _periods()
    if value is None or value not in avail:
        return DEFAULT_PERIOD if DEFAULT_PERIOD in avail else (
            avail[0] if avail else DEFAULT_PERIOD)
    return value


def _sl(value: float | None) -> float:
    """요청된 손절 기준을 DB에 저장된 값 중 하나로 정규화."""
    if value is None:
        return DEFAULT_SL
    for lvl in STOP_LOSS_LEVELS:
        if abs(lvl - float(value)) < 1e-9:
            return lvl
    raise HTTPException(400, f"지원하지 않는 손절 기준: {value} "
                             f"(가능: {', '.join(f'{x:g}' for x in STOP_LOSS_LEVELS)})")


def _refresh_info() -> dict:
    """마지막 데이터 갱신 시각 + 다음 예약 실행 시각."""
    m = all_meta()
    jobs = []
    if scheduler:
        jobs = [{"id": j.id,
                 "next_run": j.next_run_time.isoformat(timespec="seconds")
                 if j.next_run_time else None}
                for j in scheduler.get_jobs()]
    return {
        "last_backtest_at": m.get("last_backtest_at"),      # UTC ISO8601
        "last_backtest_trigger": m.get("last_backtest_trigger"),
        "last_backtest_rows": m.get("last_backtest_rows"),
        "last_backtest_seconds": m.get("last_backtest_seconds"),
        "last_price_date": m.get("last_price_date"),
        "last_error": m.get("last_backtest_error"),
        "running": _job_lock.locked(),
        "timezone": TZ,
        "scheduler_enabled": SCHEDULER_ENABLED,
        "jobs": jobs,
    }


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/status")
def status():
    """헬스체크 + 갱신 상태 (업타임 모니터용)."""
    return {"ok": True, "has_results": has_results(), **_refresh_info()}


@app.post("/api/refresh-backtest")
def refresh_backtest():
    """백테스트 수동 재실행 (백그라운드)."""
    if _job_lock.locked():
        raise HTTPException(409, "이미 실행 중입니다.")
    threading.Thread(target=_run_backtests, args=("manual",), daemon=True).start()
    return {"started": True}


@app.get("/api/meta")
def meta(sl: float | None = None, period: str | None = None):
    sl, period = _sl(sl), _period(period)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT strategy_name, ticker FROM backtest_results "
            "WHERE stop_loss_pct=? AND period=? ORDER BY strategy_name, ticker",
            (sl, period)).fetchall()
        prange = conn.execute(
            "SELECT MIN(start_date) a, MAX(end_date) b FROM backtest_results "
            "WHERE period=?", (period,)).fetchone()
    order = list(strategies.REGISTRY)
    names = sorted({r["strategy_name"] for r in rows},
                   key=lambda n: order.index(n) if n in order else 99)
    return {
        "stop_loss_pct": sl,
        "stop_loss_levels": STOP_LOSS_LEVELS,
        "default_stop_loss_pct": DEFAULT_SL,
        "refresh": _refresh_info(),
        "period": period,
        "period_range": {"start": prange["a"], "end": prange["b"]} if prange else None,
        "periods": [{"name": p, **PERIOD_META.get(p, {"label": p, "note": ""})}
                    for p in _periods()],
        "strategies": [{
            "name": n,
            "label": STRATEGY_LABELS.get(n, n),
            "desc": strategies.get(n, sl).desc if n in strategies.REGISTRY else "",
        } for n in names],
        "tickers_by_strategy": {
            n: [r["ticker"] for r in rows if r["strategy_name"] == n] for n in names},
    }


@app.get("/api/result")
def result(strategy: str, ticker: str, sl: float | None = None,
           period: str | None = None):
    sl, period = _sl(sl), _period(period)
    with get_conn() as conn:
        res = conn.execute(
            "SELECT * FROM backtest_results "
            "WHERE strategy_name=? AND ticker=? AND stop_loss_pct=? AND period=?",
            (strategy, ticker, sl, period)).fetchone()
        if res is None:
            raise HTTPException(404, "결과 없음. run_backtest.py를 먼저 실행하세요.")
        trades = [dict(r) for r in conn.execute(
            "SELECT * FROM trade_logs WHERE strategy_name=? AND ticker=? "
            "AND stop_loss_pct=? AND period=? ORDER BY entry_date",
            (strategy, ticker, sl, period)).fetchall()]

    summary = dict(res)
    equity = json.loads(summary.pop("equity_curve_json"))

    # 지표 워밍업 확보 후 표시 구간만 슬라이스
    warm = (pd.Timestamp(summary["start_date"]) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    df = fetch_ohlcv(ticker, warm, summary["end_date"])
    ind = strategies.chart_indicators(df).loc[summary["start_date"]:summary["end_date"]]
    col = lambda c: [None if v != v else round(v, 2) for v in ind[c]]
    price = {
        "labels": [d.strftime("%Y-%m-%d") for d in ind.index],
        "close": col("close"), "sma20": col("sma20"), "sma50": col("sma50"),
        "bb_upper": col("bb_upper"), "bb_lower": col("bb_lower"),
    }
    return {"summary": summary, "equity": equity, "price": price, "trades": trades}


@app.get("/api/compare")
def compare(ticker: str, sl: float | None = None, period: str | None = None):
    sl, period = _sl(sl), _period(period)
    order = list(strategies.REGISTRY)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT strategy_name, total_return_pct, cagr_pct, mdd_pct, win_rate_pct, "
            "total_trades, sharpe_ratio, sortino_ratio, calmar_ratio, "
            "max_dd_recovery_days, dd_recovery_ongoing FROM backtest_results "
            "WHERE ticker=? AND stop_loss_pct=? AND period=?",
            (ticker, sl, period)).fetchall()
    bm = set(strategies.BENCHMARKS)
    out = [dict(r) | {"label": STRATEGY_LABELS.get(r["strategy_name"], r["strategy_name"]),
                      "benchmark": r["strategy_name"] in bm}
           for r in rows]
    out.sort(key=lambda r: order.index(r["strategy_name"])
             if r["strategy_name"] in order else 99)
    return out


@app.get("/api/matrix")
def matrix(sl: float | None = None, period: str | None = None):
    """종목(행) x 전략(열) 교차 매트릭스 + 평균 요약."""
    sl, period = _sl(sl), _period(period)
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT strategy_name, ticker, total_return_pct, total_trades, "
            "mdd_pct, win_rate_pct, sharpe_ratio FROM backtest_results "
            "WHERE stop_loss_pct=? AND period=?", (sl, period)).fetchall()]

    order = list(strategies.REGISTRY)
    names = sorted({r["strategy_name"] for r in rows},
                   key=lambda n: order.index(n) if n in order else 99)
    tickers = sorted({r["ticker"] for r in rows})
    cells = {(r["strategy_name"], r["ticker"]): r for r in rows}

    # 벤치마크(Buy&Hold)는 평균 집계에서 제외한다
    bm = set(strategies.BENCHMARKS)
    trading = [n for n in names if n not in bm]

    def avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "stop_loss_pct": sl, "period": period,
        "strategies": [{"name": n, "label": STRATEGY_LABELS.get(n, n),
                        "benchmark": n in bm} for n in names],
        "tickers": tickers,
        "cells": {t: [cells.get((n, t)) for n in names] for t in tickers},
        "ticker_avg": {t: avg([cells[(n, t)]["total_return_pct"]
                               for n in trading if (n, t) in cells]) for t in tickers},
        "strategy_avg": {n: avg([cells[(n, t)]["total_return_pct"]
                                 for t in tickers if (n, t) in cells]) for n in names},
        "overall_avg": avg([r["total_return_pct"] for r in rows
                            if r["strategy_name"] not in bm]),
    }


# '전체 평균 2' 에서 제외할 (전략, 종목) 조합 — 수익률이 과도하게 큰 이상치
EXCLUDED_FROM_OVERALL2 = {("Gemini_2", "NVDA")}


@app.get("/api/stoploss-comparison")
def stoploss_comparison(period: str | None = None):
    """전략 x 종목 x 손절기준 종합 비교 매트릭스."""
    period = _period(period)
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT strategy_name, ticker, stop_loss_pct, total_return_pct, mdd_pct, "
            "win_rate_pct, total_trades, sharpe_ratio FROM backtest_results "
            "WHERE period=?", (period,)).fetchall()]
    if not rows:
        raise HTTPException(404, "결과 없음. run_backtest.py를 먼저 실행하세요.")

    order = list(strategies.REGISTRY)
    bm = set(strategies.BENCHMARKS)
    all_names = sorted({r["strategy_name"] for r in rows},
                       key=lambda n: order.index(n) if n in order else 99)
    names = [n for n in all_names if n not in bm]        # 집계 대상 = 전략만
    bench_names = [n for n in all_names if n in bm]
    tickers = sorted({r["ticker"] for r in rows})
    levels = sorted({r["stop_loss_pct"] for r in rows})
    cell = {(r["strategy_name"], r["ticker"], r["stop_loss_pct"]): r for r in rows}
    rows = [r for r in rows if r["strategy_name"] not in bm]   # 평균에서 벤치마크 제외

    def agg(subset):
        n = len(subset)
        if not n:
            return None
        return {
            "total_return_pct": round(sum(r["total_return_pct"] for r in subset) / n, 2),
            "mdd_pct": round(sum(r["mdd_pct"] for r in subset) / n, 2),
            "sharpe_ratio": round(sum(r["sharpe_ratio"] for r in subset) / n, 3),
            "win_rate_pct": round(sum(r["win_rate_pct"] for r in subset) / n, 2),
            "total_trades": sum(r["total_trades"] for r in subset),
        }

    # 전략 x 종목 행, 손절 기준 열
    grid = []
    for n in names + bench_names:
        for t in tickers:
            cells = [cell.get((n, t, lv)) for lv in levels]
            valid = [(lv, c) for lv, c in zip(levels, cells) if c]
            grid.append({
                "strategy": n, "label": STRATEGY_LABELS.get(n, n), "ticker": t,
                "benchmark": n in bm, "cells": cells,
                "best_return_sl": max(valid, key=lambda x: x[1]["total_return_pct"])[0] if valid else None,
                "best_mdd_sl": max(valid, key=lambda x: x[1]["mdd_pct"])[0] if valid else None,
            })

    strategy_rows = []
    for n in names + bench_names:
        cells = [agg([cell[(n, t, lv)] for t in tickers if (n, t, lv) in cell])
                 for lv in levels]
        valid = [(lv, c) for lv, c in zip(levels, cells) if c]
        strategy_rows.append({
            "strategy": n, "label": STRATEGY_LABELS.get(n, n), "benchmark": n in bm,
            "cells": cells,
            "best_return_sl": max(valid, key=lambda x: x[1]["total_return_pct"])[0] if valid else None,
            "best_mdd_sl": max(valid, key=lambda x: x[1]["mdd_pct"])[0] if valid else None,
        })

    overall = [agg([r for r in rows if r["stop_loss_pct"] == lv]) for lv in levels]
    ov = [(lv, c) for lv, c in zip(levels, overall) if c]

    # '전체 평균 2': 특정 조합(이상치)을 제외한 평균
    excl = [r for r in rows
            if (r["strategy_name"], r["ticker"]) not in EXCLUDED_FROM_OVERALL2]
    overall2 = [agg([r for r in excl if r["stop_loss_pct"] == lv]) for lv in levels]
    ov2 = [(lv, c) for lv, c in zip(levels, overall2) if c]
    n_cells = len({(r["strategy_name"], r["ticker"]) for r in rows})
    n_cells2 = len({(r["strategy_name"], r["ticker"]) for r in excl})

    return {
        "period": period,
        "levels": levels,
        "tickers": tickers,
        "strategy_rows": strategy_rows,
        "grid": grid,
        "overall": overall,
        "overall_best_return_sl": max(ov, key=lambda x: x[1]["total_return_pct"])[0] if ov else None,
        "overall_best_mdd_sl": max(ov, key=lambda x: x[1]["mdd_pct"])[0] if ov else None,
        "overall2": overall2,
        "overall2_best_return_sl": max(ov2, key=lambda x: x[1]["total_return_pct"])[0] if ov2 else None,
        "overall2_best_mdd_sl": max(ov2, key=lambda x: x[1]["mdd_pct"])[0] if ov2 else None,
        "overall2_excluded": [{"strategy": s, "label": STRATEGY_LABELS.get(s, s), "ticker": t}
                              for s, t in EXCLUDED_FROM_OVERALL2],
        "overall_cell_count": n_cells,
        "overall2_cell_count": n_cells2,
    }


# 실전 신호 점검 설정 — 상단 손절 버튼과 무관하게 여기 값으로 고정한다
SIGNAL_STRATEGY = "SoongI_3"   # 확정 대표 전략
# 숭이 3호는 '매수 4회 소진 후' 에만 비상 손절이 발동해 1~20% 구간 결과가 동일하다
# (실측: 7종목 평균 수익률·MDD·Calmar 전부 불변). 전략 설계값인 15% 를 쓴다.
SIGNAL_SL = 15.0
SIGNAL_YEARS = 3               # 포지션 상태 재현용 재생 기간


def _tickers():
    with get_conn() as conn:
        return [r["ticker"] for r in conn.execute(
            "SELECT DISTINCT ticker FROM backtest_results ORDER BY ticker")]


def _collect_signals(force: bool = False, sl: float | None = None):
    end = date.today() + timedelta(days=1)          # yfinance end 는 exclusive
    start = date.today() - timedelta(days=365 * SIGNAL_YEARS)
    fetch_start = start - timedelta(days=400)       # EMA200 워밍업
    sl = SIGNAL_SL if sl is None else _sl(sl)   # 기본값은 신호 전용 설정
    strat = strategies.get(SIGNAL_STRATEGY, sl)

    signals, errors = [], []
    for tk in _tickers():
        try:
            df = fetch_ohlcv(tk, fetch_start.isoformat(), end.isoformat(), force=force)
            signals.append(backtest.live_signal(df, tk, SIGNAL_STRATEGY,
                                                start=start.isoformat(), stop_loss_pct=sl))
        except Exception as exc:                    # 한 종목 실패가 전체를 막지 않도록
            errors.append({"ticker": tk, "error": str(exc)})

    latest = max((s["date"] for s in signals if "date" in s), default=None)
    return {
        "strategy": {"name": SIGNAL_STRATEGY, "label": strat.label, "desc": strat.desc},
        "stop_loss_pct": sl,
        "data_date": latest,
        "checked_at": date.today().isoformat(),
        "signals": signals,
        "errors": errors,
    }


# ---------------------------------------------------------------- 파라미터 최적화
@app.get("/optimize")
def optimize_page():
    return FileResponse(BASE / "static" / "optimize.html")


@app.get("/api/optimize/meta")
def optimize_meta():
    return {
        "strategy": {"name": optimizer.STRATEGY,
                     "label": STRATEGY_LABELS.get(optimizer.STRATEGY, optimizer.STRATEGY),
                     "desc": strategies.get(optimizer.STRATEGY).desc},
        "modes": [{"name": m, "label": v["label"], "grid": v["grid"],
                   "cols": v["cols"], "total": v["total"]}
                  for m, v in optimizer.MODES.items()],
        "param_labels": optimizer.LABELS,
        "grid": optimizer.GRID,                 # 하위호환
        "total_combos": optimizer.TOTAL,
        "tickers": optimizer.TICKERS,
        "periods": list(optimizer.PERIODS),
        "stop_loss_levels": sorted(optimizer.SL_LEVELS),
        "default": {"scope": "ALL", "period": "3Y", "sl": 15.0, "mode": "basic"},
    }


OPT_METRICS = ("total_return_pct", "mdd_pct", "sharpe_ratio", "sortino_ratio",
               "cagr_pct", "calmar_ratio", "win_rate_pct", "total_trades")


@app.post("/api/optimize")
def optimize_run(scope: str = "ALL", period: str = "3Y", sl: float = 15.0,
                 mode: str = "basic", force: bool = False):
    """그리드 서치 실행 (캐시 우선). 결과 전체를 반환한다."""
    if scope != "ALL" and scope not in optimizer.TICKERS:
        raise HTTPException(400, f"지원하지 않는 종목: {scope}")
    if period not in optimizer.PERIODS:
        raise HTTPException(400, f"지원하지 않는 기간: {period}")
    if mode not in optimizer.MODES:
        raise HTTPException(400, f"지원하지 않는 모드: {mode}")
    cols = optimizer.MODES[mode]["cols"]
    t0 = time.perf_counter()
    try:
        rows, cached = optimizer.run_grid(scope, period, float(sl),
                                          force=force, mode=mode)
    except Exception as exc:                       # 연산 실패를 그대로 노출
        raise HTTPException(500, f"최적화 실행 실패: {exc}")

    def pick(r):
        return {k: r[k] for k in (*cols, *OPT_METRICS)}

    def best(key):
        return pick(max(rows, key=lambda x: x.get(key) or 0))

    return {
        "scope": scope, "period": period, "stop_loss_pct": float(sl), "mode": mode,
        "cols": cols, "cached": cached,
        "elapsed_sec": round(time.perf_counter() - t0, 2), "count": len(rows),
        "top": {
            "calmar": best("calmar_ratio"),
            "return": best("total_return_pct"),
            "mdd": best("mdd_pct"),          # mdd 는 음수 — 큰 값이 얕은 낙폭
        },
        "rows": [pick(r) for r in rows],
    }


@app.get("/api/today-signals")
def today_signals(sl: float | None = None):
    """캐시된 최신 일봉 기준 오늘의 신호."""
    return _collect_signals(force=False, sl=sl)


@app.post("/api/refresh-signals")
def refresh_signals(sl: float | None = None):
    """yfinance 에서 최신 주가를 다시 받아 DB 갱신 후 신호 재계산."""
    return _collect_signals(force=True, sl=sl)


if __name__ == "__main__":
    import uvicorn
    # 클라우드(Render 등)는 $PORT 를 주고 0.0.0.0 바인딩을 요구한다
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")))
