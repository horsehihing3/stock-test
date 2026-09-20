"""데이터 수집 -> 전 전략 x 전 종목 백테스트 -> DB 저장 -> 비교표 출력."""
from datetime import date, datetime, timedelta, timezone

import backtest
import strategies
from data_loader import fetch_ohlcv
from database import init_db, purge_unregistered, save_backtest, set_meta

TICKERS = ["BTC-USD", "NVDA", "AAPL", "QQQ", "SPY", "SLV", "SOXX"]
# 대표 전략(숭이 3호)을 맨 앞에 둔다 — 대시보드 드롭다운/매트릭스 순서가 이 순서를 따른다
STRATEGIES = ["SoongI_3", "Gemini_2", "Claude_2", "Gemini_4", "SoongI_1", "SoongI_2"]
BENCHMARKS = list(strategies.BENCHMARKS)               # ["Buy_Hold"] — 집계 제외
ALL_RUNS = STRATEGIES + BENCHMARKS

STOP_LOSS_LEVELS = list(strategies.STOP_LOSS_LEVELS)   # (3, 5, 8, 10, 15, 20, 25, 30) %
DEFAULT_SL = strategies.DEFAULT_STOP_LOSS_PCT          # 상세표에 쓸 기준 손절

END = date.today()

# 기간 정의 — DB 에 period 라벨로 구분 저장되어 두 구간을 나란히 볼 수 있다
PERIODS = {
    "1Y":   {"label": "최근 1년",
             "start": END - timedelta(days=365), "end": END,
             "note": "최근 12개월 · 표본이 짧아 순위는 노이즈로 봐야 함"},
    "3Y":   {"label": "최근 3년 (상승장)",
             "start": END - timedelta(days=365 * 3), "end": END,
             "note": "2023-09~2026-09 · 4종목 모두 상승"},
    "5.7Y": {"label": "2021-01~현재 (하락장 포함)",
             "start": date(2021, 1, 1), "end": END,
             "note": "2021-11 고점 → 2022 하락 → 2023~2026 상승"},
}
DEFAULT_PERIOD = "3Y"

# 가장 이른 시작일 기준으로 워밍업 400일 확보 (2021-01-01 - 400d ≈ 2019-11)
FETCH_START = date(2019, 10, 1)


def bar(title, width=100):
    print(f"\n{title}\n" + "=" * width)


def cagr(res):
    """연평균 복리 수익률(%). 엔진이 계산한 값을 그대로 쓴다."""
    return res.get("cagr_pct", 0.0)


def rec(res):
    """최장 드로다운 회복 기간 표기 (미회복이면 '+')."""
    return f"{res['max_dd_recovery_days']}{'+' if res['dd_recovery_ongoing'] else ''}"


def run_period(pkey, prices, start_override=None):
    """한 기간에 대해 손절 x 전략 x 종목 전부 실행하고 DB 에 저장."""
    p = PERIODS[pkey]
    s = (start_override or p["start"]).isoformat()
    all_results = {}     # (sl, sname, tk) -> result
    for sl in STOP_LOSS_LEVELS:
        for sname in ALL_RUNS:
            for tk in TICKERS:
                res, trades, _ = backtest.run(
                    prices[tk], tk, sname, start=s, end=p["end"].isoformat(),
                    stop_loss_pct=sl, period=pkey)
                save_backtest(res, trades)
                all_results[(sl, sname, tk)] = res
    print(f"  [{pkey}] {s} ~ {p['end']}  {p['label']} -> {len(all_results)}건")
    return all_results


def report(pkey, all_results):
    p = PERIODS[pkey]
    results = {(s, t): all_results[(DEFAULT_SL, s, t)] for s in ALL_RUNS for t in TICKERS}
    bar(f"■■■ 기간 [{pkey}] {p['label']}  —  {p['start']} ~ {p['end']}  ({p['note']})")

    # ---------------- 전략별 상세 (기준 손절 = DEFAULT_SL) ----------------
    for sname in ALL_RUNS:
        s = strategies.get(sname, DEFAULT_SL)
        bar(f"[{s.label}] {s.name}  -  {s.desc}")
        print(f"{'TICKER':<10}{'RETURN%':>10}{'CAGR%':>9}{'MDD%':>9}{'WIN%':>8}"
              f"{'TRADES':>8}{'SHARPE':>9}{'SORTINO':>9}{'CALMAR':>8}{'회복일':>8}"
              f"{'최종자산($)':>14}")
        print("-" * 100)
        for tk in TICKERS:
            r = results[(sname, tk)]
            print(f"{tk:<10}{r['total_return_pct']:>10.2f}{cagr(r):>9.2f}{r['mdd_pct']:>9.2f}"
                  f"{r['win_rate_pct']:>8.1f}{r['total_trades']:>8}"
                  f"{r['sharpe_ratio']:>9.3f}{r['sortino_ratio']:>9.3f}"
                  f"{r['calmar_ratio']:>8.2f}{rec(r):>8}{r['final_equity']:>14,.2f}")
        rs = [results[(sname, t)] for t in TICKERS]
        n = len(rs)
        print("-" * 100)
        print(f"{'평균':<10}{sum(r['total_return_pct'] for r in rs)/n:>10.2f}"
              f"{sum(cagr(r) for r in rs)/n:>9.2f}"
              f"{sum(r['mdd_pct'] for r in rs)/n:>9.2f}"
              f"{sum(r['win_rate_pct'] for r in rs)/n:>8.1f}"
              f"{sum(r['total_trades'] for r in rs):>8}"
              f"{sum(r['sharpe_ratio'] for r in rs)/n:>9.3f}"
              f"{sum(r['sortino_ratio'] for r in rs)/n:>9.3f}"
              f"{sum(r['calmar_ratio'] for r in rs)/n:>8.2f}"
              f"{sum(r['max_dd_recovery_days'] for r in rs)/n:>8.0f}")

    # ---------------- 종목 x 전략 매트릭스 (종목=행, 전략=열) ----------------
    bar("종목 x 전략 매트릭스 - 총수익률(%) / 거래횟수")
    print(f"{'종목':<10}" + "".join(f"{strategies.get(s).label:>18}" for s in STRATEGIES)
          + f"{'종목평균':>14}" + f"{'  |':>3}" + f"{'B&H':>16}")
    print("-" * 100)
    for tk in TICKERS:
        line = f"{tk:<10}"
        for sname in STRATEGIES:
            r = results[(sname, tk)]
            line += f"{r['total_return_pct']:>12.2f}% /{r['total_trades']:>3}"
        avg = sum(results[(s, tk)]['total_return_pct'] for s in STRATEGIES) / len(STRATEGIES)
        bh = results[(BENCHMARKS[0], tk)]['total_return_pct'] if BENCHMARKS else 0.0
        print(line + f"{avg:>13.2f}%" + "   |" + f"{bh:>15.2f}%")
    print("-" * 100)
    line = f"{'전략평균':<10}"
    for sname in STRATEGIES:
        avg = sum(results[(sname, t)]['total_return_pct'] for t in TICKERS) / len(TICKERS)
        line += f"{avg:>12.2f}%    "
    overall = (sum(results[(s, t)]['total_return_pct'] for s in STRATEGIES for t in TICKERS)
               / (len(STRATEGIES) * len(TICKERS)))
    bh_avg = (sum(results[(BENCHMARKS[0], t)]['total_return_pct'] for t in TICKERS)
              / len(TICKERS)) if BENCHMARKS else 0.0
    print(line + f"{overall:>13.2f}%" + "   |" + f"{bh_avg:>15.2f}%")

    # ---------------- Buy&Hold 대비 초과성과 (종목=행) ----------------
    for bname in BENCHMARKS:
        bm = strategies.get(bname)
        bar(f"{bm.label} 대비 초과성과(%p)  [손절 {DEFAULT_SL:g}% 기준]")
        print(f"{'종목':<10}" + "".join(f"{strategies.get(s).label:>16}" for s in STRATEGIES)
              + f"{'B&H 수익률':>16}")
        print("-" * 100)
        wins = {s: 0 for s in STRATEGIES}
        for tk in TICKERS:
            line, base = f"{tk:<10}", results[(bname, tk)]['total_return_pct']
            for sname in STRATEGIES:
                diff = results[(sname, tk)]['total_return_pct'] - base
                wins[sname] += diff > 0
                line += f"{diff:>+14.2f}  "
            print(line + f"{base:>14.2f}%")
        print("-" * 100)
        print(f"{'승/패':<10}" + "".join(
            f"{f'{wins[s]}승 {len(TICKERS)-wins[s]}패':>16}" for s in STRATEGIES))

    # ---------------- 종합 랭킹 ----------------
    bar(f"종합 랭킹 ({len(TICKERS)}종목 평균) — 벤치마크 포함 비교")
    print(f"{'순위':<6}{'전략':<14}{'수익률%':>11}{'CAGR%':>9}{'MDD%':>9}"
          f"{'Sharpe':>9}{'Sortino':>9}{'Calmar':>8}{'최장회복일':>11}")
    print("-" * 100)
    rank = []
    for sname in ALL_RUNS:
        rs = [results[(sname, t)] for t in TICKERS]
        n = len(rs)
        rank.append((sum(r['total_return_pct'] for r in rs) / n,
                     sum(cagr(r) for r in rs) / n,
                     sum(r['mdd_pct'] for r in rs) / n,
                     sum(r['sharpe_ratio'] for r in rs) / n,
                     sum(r['sortino_ratio'] for r in rs) / n,
                     sum(r['calmar_ratio'] for r in rs) / n,
                     sum(r['max_dd_recovery_days'] for r in rs) / n,
                     sname in BENCHMARKS, strategies.get(sname).label))
    order = sorted([x for x in rank if not x[7]], reverse=True) + [x for x in rank if x[7]]
    for i, (ret, cg, mdd, shp, srt, clm, rcv, bench, label) in enumerate(order, 1):
        tag = "  (벤치마크)" if bench else ""
        print(f"{('-' if bench else i):<6}{label:<14}{ret:>11.2f}{cg:>9.2f}{mdd:>9.2f}"
              f"{shp:>9.3f}{srt:>9.3f}{clm:>8.2f}{rcv:>11.0f}{tag}")

    # ---------------- 손절 기준별 비교 ----------------
    def avg_of(sl, sname=None):
        rs = [all_results[(sl, s, t)] for s in ([sname] if sname else STRATEGIES)
              for t in TICKERS]
        n = len(rs)
        return (sum(r['total_return_pct'] for r in rs) / n,
                sum(r['mdd_pct'] for r in rs) / n,
                sum(r['sharpe_ratio'] for r in rs) / n)

    stats = {(sl, s): avg_of(sl, s) for sl in STOP_LOSS_LEVELS for s in STRATEGIES}
    overall = {sl: avg_of(sl) for sl in STOP_LOSS_LEVELS}

    for idx, metric in ((0, "평균 수익률(%)"), (1, "평균 MDD(%)")):
        bar(f"손절 기준별 비교 - 손절(행) x 전략(열)  [{metric}]")
        print(f"{'손절':<8}" + "".join(f"{strategies.get(s).label:>16}" for s in STRATEGIES)
              + f"{'전체평균':>14}")
        print("-" * 100)
        for sl in STOP_LOSS_LEVELS:
            line = f"{f'{sl:g}%':<8}"
            for s in STRATEGIES:
                line += f"{stats[(sl, s)][idx]:>14.2f}  "
            print(line + f"{overall[sl][idx]:>12.2f}")
        print("-" * 100)
        line = f"{'최적':<8}"
        for s in STRATEGIES:
            best = (max if idx == 0 else max)(STOP_LOSS_LEVELS,
                                              key=lambda lv: stats[(lv, s)][idx])
            line += f"{f'{best:g}%':>14}  "
        best_o = max(STOP_LOSS_LEVELS, key=lambda lv: overall[lv][idx])
        print(line + f"{f'{best_o:g}%':>12}")


def cross_period(runs):
    """기간을 나란히 대조. 상세 대조는 최단 구간 vs 최장 구간으로 한다."""
    keys = list(PERIODS)
    res = {k: {(s, t): runs[k][(DEFAULT_SL, s, t)] for s in ALL_RUNS for t in TICKERS}
           for k in keys}

    # --- 기간별 요약 (전 기간 한눈에) ---
    bar(f"■■■ 기간별 요약 — 4종목 평균 [손절 {DEFAULT_SL:g}% 기준]")
    print(f"{'전략':<14}" + "".join(f"{f'{k} 수익률':>12}{f'{k} MDD':>10}{f'{k} Cal':>9}"
                                    for k in keys))
    print("-" * 100)
    for sname in ALL_RUNS:
        line = f"{strategies.get(sname).label:<14}"
        for k in keys:
            rs = [res[k][(sname, t)] for t in TICKERS]
            n = len(rs)
            line += (f"{sum(r['total_return_pct'] for r in rs)/n:>12.2f}"
                     f"{sum(r['mdd_pct'] for r in rs)/n:>10.2f}"
                     f"{sum(r['calmar_ratio'] for r in rs)/n:>9.2f}")
        print(line)

    a, b = keys[0], keys[-1]

    bar(f"■■■ 구간 대조  [{a}] vs [{b}]   수익률% (MDD%)   손절 {DEFAULT_SL:g}% 기준")
    print(f"{'전략':<13}{'종목':<9}{f'{a} 수익률':>12}{f'{a} MDD':>10}"
          f"{f'{b} 수익률':>13}{f'{b} MDD':>10}{'B&H초과 ' + a:>14}{'B&H초과 ' + b:>14}")
    print("-" * 100)
    flips = []
    for sname in ALL_RUNS:
        for tk in TICKERS:
            ra, rb = res[a][(sname, tk)], res[b][(sname, tk)]
            ba, bb = res[a][(BENCHMARKS[0], tk)], res[b][(BENCHMARKS[0], tk)]
            xa = ra['total_return_pct'] - ba['total_return_pct']
            xb = rb['total_return_pct'] - bb['total_return_pct']
            bench = sname in BENCHMARKS
            print(f"{strategies.get(sname).label:<13}{tk:<9}"
                  f"{ra['total_return_pct']:>12.2f}{ra['mdd_pct']:>10.2f}"
                  f"{rb['total_return_pct']:>13.2f}{rb['mdd_pct']:>10.2f}"
                  f"{'  -' if bench else f'{xa:>+14.2f}'}"
                  f"{'  -' if bench else f'{xb:>+14.2f}'}")
            if not bench and (xa > 0) != (xb > 0):
                flips.append((strategies.get(sname).label, tk, xa, xb))
        print("-" * 100)

    bar("B&H 대비 승패 변화")
    for k in keys:
        wins = sum(1 for s in STRATEGIES for t in TICKERS
                   if res[k][(s, t)]['total_return_pct']
                   > res[k][(BENCHMARKS[0], t)]['total_return_pct'])
        total = len(STRATEGIES) * len(TICKERS)
        print(f"  [{k}] {PERIODS[k]['label']}: {wins}승 {total - wins}패 / {total}조합")
    print(f"\n  승패가 뒤집힌 조합: {len(flips)}개")
    for label, tk, xa, xb in flips:
        print(f"    {label:<13}{tk:<9}{a} {xa:>+9.2f}%p  ->  {b} {xb:>+9.2f}%p")


def run_all_backtests(report_output=True, trigger="manual"):
    """스케줄러/부트스트랩용 진입점. 데이터 수집 + 전 기간 백테스트 + DB 저장.

    report_output=False 면 터미널 리포트를 생략하고 요약만 남긴다.
    마지막 갱신 시각은 app_meta 에 기록해 재시작 후에도 유지된다.
    """
    started = datetime.now(timezone.utc)
    init_db()
    purge_unregistered(ALL_RUNS)
    prices = {tk: fetch_ohlcv(tk, FETCH_START.isoformat(), END.isoformat())
              for tk in TICKERS}
    runs = {k: run_period(k, prices) for k in PERIODS}

    if report_output:
        for k in PERIODS:
            report(k, runs[k])
        cross_period(runs)

    total = sum(len(v) for v in runs.values())
    took = (datetime.now(timezone.utc) - started).total_seconds()
    last_bar = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    set_meta("last_backtest_at", started.isoformat(timespec="seconds"))
    set_meta("last_backtest_trigger", trigger)
    set_meta("last_backtest_rows", total)
    set_meta("last_backtest_seconds", round(took, 1))
    set_meta("last_price_date", last_bar)
    return {"rows": total, "seconds": round(took, 1),
            "last_price_date": last_bar, "trigger": trigger}


def main():
    init_db()
    stale = purge_unregistered(ALL_RUNS)
    if stale:
        print(f"폐기 전략 결과 제거: {', '.join(stale)}")
    print(f"데이터 수집: {FETCH_START} ~ {END} (워밍업 포함)")
    prices = {tk: fetch_ohlcv(tk, FETCH_START.isoformat(), END.isoformat())
              for tk in TICKERS}
    for tk, df in prices.items():
        print(f"  {tk:<9}{len(df):>6}행  {df.index[0].date()} ~ {df.index[-1].date()}")

    print("\n백테스트 실행:")
    runs = {k: run_period(k, prices) for k in PERIODS}

    for k in PERIODS:
        report(k, runs[k])
    cross_period(runs)

    total = sum(len(v) for v in runs.values())
    print(f"\n완료: 기간 {len(PERIODS)}종 x 손절 {len(STOP_LOSS_LEVELS)}종 x "
          f"(전략 {len(STRATEGIES)} + 벤치마크 {len(BENCHMARKS)}) x 종목 {len(TICKERS)}개 "
          f"= {total}건을 market_data.db에 저장")


if __name__ == "__main__":
    main()
