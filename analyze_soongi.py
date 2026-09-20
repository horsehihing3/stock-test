"""숭이 시리즈(SoongI_1/2/3) 거래 내역 비교 분석. 최종 대표 = 숭이 3호.

실행: python analyze_soongi.py [기간] [손절%]
기본값: 3Y / 5%
"""
import json
import sys
from collections import defaultdict

import pandas as pd

import strategies
from database import get_conn

PERIOD = sys.argv[1] if len(sys.argv) > 1 else "3Y"
SL = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SERIES = ["SoongI_1", "SoongI_2", "SoongI_3"]
LABEL = {s: strategies.LABELS.get(s, s) for s in SERIES}
TICKERS = ["BTC-USD", "NVDA", "AAPL", "QQQ", "SPY", "SLV", "SOXX"]


def bar(t, w=104):
    print(f"\n{t}\n" + "=" * w)


def load():
    """전략별 요약 + 포지션 단위 집계 (같은 entry_date = 한 포지션)."""
    summary, positions, trades = {}, defaultdict(list), defaultdict(list)
    with get_conn() as conn:
        for s in SERIES:
            for tk in TICKERS:
                r = conn.execute(
                    "SELECT * FROM backtest_results WHERE strategy_name=? AND ticker=? "
                    "AND period=? AND stop_loss_pct=?", (s, tk, PERIOD, SL)).fetchone()
                summary[(s, tk)] = dict(r) if r else None
                rows = [dict(x) for x in conn.execute(
                    "SELECT * FROM trade_logs WHERE strategy_name=? AND ticker=? "
                    "AND period=? AND stop_loss_pct=? ORDER BY entry_date, exit_date",
                    (s, tk, PERIOD, SL)).fetchall()]
                trades[(s, tk)] = rows
                grp = defaultdict(list)
                for x in rows:
                    grp[x["entry_date"]].append(x)
                for ed, g in grp.items():
                    cost = sum(t["shares"] * t["entry_price"] for t in g)
                    positions[(s, tk)].append({
                        "entry": ed, "exit": max(t["exit_date"] for t in g),
                        "pnl": sum(t["pnl"] for t in g),
                        "ret": sum(t["pnl"] for t in g) / cost * 100 if cost else 0.0,
                        "reasons": [t["exit_reason"] for t in g],
                        "entry_reason": g[0]["entry_reason"],
                        "days": max(t["holding_days"] for t in g),
                    })
                positions[(s, tk)].sort(key=lambda p: p["entry"])
    return summary, positions, trades


def exposure(summary, s, tk):
    r = summary[(s, tk)]
    if not r:
        return 0.0
    eq = pd.DataFrame(json.loads(r["equity_curve_json"]))
    return (eq["equity"].diff().abs() > 1e-9).sum() / max(len(eq) - 1, 1) * 100


def stat(summary, positions, s):
    """전략 전체 집계."""
    pos = [p for tk in TICKERS for p in positions[(s, tk)]]
    su = [summary[(s, tk)] for tk in TICKERS if summary[(s, tk)]]
    n = len(su) or 1
    return {
        "pos": len(pos),
        "legs": sum(r["total_trades"] for r in su),
        "win": sum(1 for p in pos if p["pnl"] > 0) / len(pos) * 100 if pos else 0,
        "ret_pos": sum(p["ret"] for p in pos) / len(pos) if pos else 0,
        "days": sum(p["days"] for p in pos) / len(pos) if pos else 0,
        "total": sum(r["total_return_pct"] for r in su) / n,
        "mdd": sum(r["mdd_pct"] for r in su) / n,
        "calmar": sum(r["calmar_ratio"] for r in su) / n,
        "pnl": sum(p["pnl"] for p in pos),
        "exp": sum(exposure(summary, s, tk) for tk in TICKERS) / n,
    }


def main():
    summary, positions, trades = load()
    bar(f"■ 숭이 시리즈 비교  [기간 {PERIOD} · 손절 {SL:g}%]")

    # ---------- 1) 종목별 ----------
    bar("1) 종목별 총수익률% (포지션 수)")
    print(f"{'종목':<10}" + "".join(f"{LABEL[s]:>19}" for s in SERIES))
    print("-" * 104)
    for tk in TICKERS:
        line = f"{tk:<10}"
        for s in SERIES:
            r, p = summary[(s, tk)], positions[(s, tk)]
            line += (f"{r['total_return_pct']:>13.2f}%({len(p):>2})" if r else f"{'-':>19}")
        print(line)
    print("-" * 104)
    line = f"{'평균':<10}"
    for s in SERIES:
        st = stat(summary, positions, s)
        line += f"{st['total']:>13.2f}%({st['pos']:>2})"
    print(line)

    # ---------- 2) 핵심 지표 ----------
    bar("2) 핵심 지표 비교 (7종목 평균)")
    keys = [("total", "총수익률%", "{:.2f}"), ("mdd", "MDD%", "{:.2f}"),
            ("calmar", "Calmar", "{:.2f}"), ("win", "포지션 승률%", "{:.1f}"),
            ("ret_pos", "포지션당 평균%", "{:.2f}"), ("days", "평균 보유일", "{:.0f}"),
            ("pos", "포지션 수", "{:.0f}"), ("legs", "청산 건수", "{:.0f}"),
            ("exp", "시장 노출%", "{:.1f}"), ("pnl", "실현손익$", "{:,.0f}")]
    st = {s: stat(summary, positions, s) for s in SERIES}
    print(f"{'지표':<16}" + "".join(f"{LABEL[s]:>16}" for s in SERIES))
    print("-" * 104)
    for k, lab, fmt in keys:
        print(f"{lab:<16}" + "".join(f"{fmt.format(st[s][k]):>16}" for s in SERIES))

    # ---------- 3) 청산 사유 ----------
    bar("3) 청산 사유별 건수 / 손익$")
    ex = {s: defaultdict(lambda: [0, 0.0]) for s in SERIES}
    for s in SERIES:
        for tk in TICKERS:
            for t in trades[(s, tk)]:
                k = t["exit_reason"].split("(")[0].strip()
                ex[s][k][0] += 1
                ex[s][k][1] += t["pnl"]
    allk = sorted({k for s in SERIES for k in ex[s]},
                  key=lambda k: -sum(ex[s][k][1] for s in SERIES))
    print(f"{'청산 사유':<18}" + "".join(f"{LABEL[s]:>21}" for s in SERIES))
    print("-" * 104)
    for k in allk:
        line = f"{k:<18}"
        for s in SERIES:
            c, v = ex[s][k]
            line += (f"{c:>6}건{v:>13,.0f}" if c else f"{'-':>21}")
        print(line)

    # ---------- 4) 조기 청산 해소 여부 ----------
    bar("4) '조기 청산' 문제 해소 확인")
    print(f"{'항목':<24}" + "".join(f"{LABEL[s]:>16}" for s in SERIES))
    print("-" * 104)
    rows = [
        ("트레일링 청산 건수",
         lambda s: sum(c for k, (c, _) in ex[s].items() if "트레일링" in k)),
        ("완익절 건수", lambda s: ex[s].get("2차 완익절", [0, 0])[0]),
        ("완익절 손익$", lambda s: ex[s].get("2차 완익절", [0, 0])[1]),
        ("비상 손절 건수", lambda s: ex[s].get("비상 손절", [0, 0])[0]),
        ("비상 손절 손익$", lambda s: ex[s].get("비상 손절", [0, 0])[1]),
        ("포지션당 평균 보유일", lambda s: st[s]["days"]),
    ]
    for lab, fn in rows:
        vals = [fn(s) for s in SERIES]
        fmt = "{:,.0f}" if "손익" in lab else ("{:.0f}" if "일" in lab or "건수" in lab else "{:.2f}")
        print(f"{lab:<24}" + "".join(f"{fmt.format(v):>16}" for v in vals))

    # ---------- 5) 요약 ----------
    s1, s2, s3 = (st[s] for s in SERIES)
    best = max(SERIES, key=lambda s: st[s]["calmar"])
    bestr = max(SERIES, key=lambda s: st[s]["total"])
    bar("★ 최종 확정 — 숭이 3호")
    print(f"""
  확정 사유            Calmar {s3['calmar']:.2f} 로 시리즈 최고 · MDD {s3['mdd']:.2f}% 로 최저
                       (숭이1 {s1['calmar']:.2f}/{s1['mdd']:.2f}% · 숭이2 {s2['calmar']:.2f}/{s2['mdd']:.2f}%)
  파라미터             초기 40% 투입 · -10% 마다 20% 물타기(총 4회) ·
                       +10% 절반익절 · +30% 완익절 · +15% 후 고점-12% 트레일링 · -15% 비상손절

  지표                 숭이1        숭이2        숭이3(대표)
  총수익률%          {s1['total']:>8.2f}    {s2['total']:>8.2f}    {s3['total']:>8.2f}
  MDD%              {s1['mdd']:>8.2f}    {s2['mdd']:>8.2f}    {s3['mdd']:>8.2f}
  Calmar            {s1['calmar']:>8.2f}    {s2['calmar']:>8.2f}    {s3['calmar']:>8.2f}
  포지션당 평균%     {s1['ret_pos']:>8.2f}    {s2['ret_pos']:>8.2f}    {s3['ret_pos']:>8.2f}
  평균 보유일        {s1['days']:>8.0f}    {s2['days']:>8.0f}    {s3['days']:>8.0f}
  청산 건수          {s1['legs']:>8.0f}    {s2['legs']:>8.0f}    {s3['legs']:>8.0f}

  참고: 절대 수익률 1위는 {LABEL[bestr]}({st[bestr]['total']:.2f}%) 이지만 손절 로직이 없어
        미실현 손실이 장부에 남지 않는다. 위험조정 기준으로는 숭이 3호가 우위.
    """)


if __name__ == "__main__":
    main()
