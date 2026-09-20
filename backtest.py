"""단일 종목/단일 전략 백테스트 엔진 (전량 매수·전량 청산, 종가 체결)."""
import json

import numpy as np
import pandas as pd

import strategies

INITIAL_CAPITAL = 10_000.0
FEE = 0.001  # 진입/청산 각각 0.1%

def live_signal(df: pd.DataFrame, ticker: str, strategy_name: str,
                start: str | None = None, stop_loss_pct: float | None = None) -> dict:
    """최신 봉 기준 '오늘의 신호'를 판정한다.

    run() 과 동일한 로직으로 과거를 재생하되 마지막 봉을 강제 청산하지 않아,
    현재 포지션 보유 여부와 오늘 발생한 신호를 그대로 반환한다.
    status: BUY | SELL | HOLD_POSITION | HOLD_FLAT
    """
    strat = strategies.get(strategy_name, stop_loss_pct)
    d = strat.indicators(df)
    if start:
        d = d.loc[start:]
    rows = list(d.itertuples())
    if not rows:
        return {"ticker": ticker, "status": "NO_DATA"}

    scale_in = getattr(strat, "scale_in", None)
    pos = None
    for i, row in enumerate(rows):
        is_last = (i == len(rows) - 1)
        if pos:
            pos["peak"] = max(pos["peak"], row.high)
            sig = strat.exit(row, pos)
            if sig:
                frac, reason = sig if isinstance(sig, tuple) else (1.0, sig)
                if is_last:
                    return _signal_row(ticker, row, "SELL", reason, pos, strat, frac)
                if frac >= 1.0:
                    pos = None
            elif scale_in:
                # 물타기 재현 — 평단가가 청산 판정에 쓰이므로 반드시 반영해야 한다.
                # (현금 한도는 모델링하지 않는다. 전략의 매수 횟수 상한만 따른다)
                add = scale_in(row, pos)
                if add:
                    q, r = pos.get("qty", 1.0), add[0]
                    pos["entry_price"] = (pos["entry_price"] * q + row.close * q * r) / (q * (1 + r))
                    pos["qty"] = q * (1 + r)
                    pos["last_buy_price"] = row.close
                    pos["add_count"] = pos.get("add_count", 0) + 1
        elif strat.entry(row):
            if is_last:
                return _signal_row(ticker, row, "BUY", strat.entry_reason(row), None, strat)
            pos = {"entry_price": row.close, "entry_date": row.Index.strftime("%Y-%m-%d"),
                   "peak": row.high, "entry_reason": strat.entry_reason(row),
                   "last_buy_price": row.close, "qty": 1.0}

    last = rows[-1]
    status = "HOLD_POSITION" if pos else "HOLD_FLAT"
    reason = "보유 유지 (청산 조건 미충족)" if pos else "진입 조건 미충족"
    return _signal_row(ticker, last, status, reason, pos, strat)


def _signal_row(ticker, row, status, reason, pos, strat, frac=None):
    out = {
        "ticker": ticker,
        "status": status,
        "date": row.Index.strftime("%Y-%m-%d"),
        "close": round(float(row.close), 2),
        "reason": reason,
        "conditions": strat.condition_report(row),
    }
    if frac is not None and frac < 1.0:
        out["exit_fraction"] = frac
    if pos:
        out["position"] = {
            "entry_date": pos["entry_date"],
            "entry_price": round(pos["entry_price"], 2),
            "unrealized_pct": round((row.close / pos["entry_price"] - 1) * 100, 2),
            "peak": round(float(pos["peak"]), 2),
            "scaled": bool(pos.get("scaled")),
        }
    return out


def run(df: pd.DataFrame, ticker: str, strategy_name: str = "Gemini_1",
        start: str | None = None, stop_loss_pct: float | None = None,
        end: str | None = None, period: str = "3Y", strat=None):
    """start/end: 지표 워밍업 이후 실제 백테스트 구간.
    stop_loss_pct: 손절선(%) 오버라이드. None 이면 전략 고유값 사용.
    period: 결과를 구분할 기간 라벨 (DB 키의 일부).
    strat: 파라미터를 직접 주입한 전략 인스턴스 (그리드 서치용)."""
    if strat is None:
        strat = strategies.get(strategy_name, stop_loss_pct)
    sl_pct = stop_loss_pct if stop_loss_pct is not None else abs(strat.SL) * 100
    d = strat.indicators(df)
    if start or end:
        d = d.loc[start:end]

    cash, shares = INITIAL_CAPITAL, 0.0
    state = {}
    trades, equity = [], []
    rows = list(d.itertuples())

    def buy(row, frac_of_cash=1.0, qty_ratio=None):
        """매수. qty_ratio 가 주어지면 '현재 보유 수량 x 비율' 만큼 추가 매수한다.
        추가 매수 시 state['entry_price'] 는 수량가중 평단가로 갱신된다."""
        nonlocal cash, shares, state
        px = row.close
        want = shares * qty_ratio if qty_ratio is not None else \
            (cash * max(0.0, min(1.0, frac_of_cash))) / (px * (1 + FEE))
        spend = want * px * (1 + FEE)
        if spend > cash:                       # 현금 부족분은 살 수 있는 만큼만
            want, spend = cash / (px * (1 + FEE)), cash
        if want <= 1e-12:
            return 0.0
        cash -= spend
        if shares > 0:
            state["entry_price"] = ((state["entry_price"] * shares + px * want)
                                    / (shares + want))
        shares += want
        state["last_buy_price"] = px
        return want

    def close_trade(row, reason, frac=1.0):
        """frac 비율만큼 청산. frac<1 이면 분할 익절(나머지 수량은 계속 보유)."""
        nonlocal cash, shares, state
        px = row.close
        qty = shares if frac >= 1.0 else shares * frac
        proceeds = qty * px * (1 - FEE)
        cost = qty * state["entry_price"] * (1 + FEE)
        ereason = state["entry_reason"]
        if state.get("add_count"):
            ereason += (f" +추가매수 {state['add_count']}회 "
                        f"→ 평단 {state['entry_price']:.2f}")
        trades.append({
            "strategy_name": strategy_name, "ticker": ticker, "stop_loss_pct": sl_pct,
            "period": period,
            "entry_date": state["entry_date"], "entry_price": round(state["entry_price"], 4),
            "exit_date": row.Index.strftime("%Y-%m-%d"), "exit_price": round(px, 4),
            "shares": round(qty, 6),
            "pnl": round(proceeds - cost, 2),
            "return_pct": round((proceeds / cost - 1) * 100, 2),
            "entry_reason": ereason,
            "exit_reason": reason,
            "holding_days": (row.Index - pd.Timestamp(state["entry_date"])).days,
        })
        cash += proceeds
        shares -= qty
        if shares <= 1e-9:
            shares, state = 0.0, {}

    scale_in = getattr(strat, "scale_in", None)

    for row in rows:
        if shares > 0:
            state["peak"] = max(state["peak"], row.high)
            sig = strat.exit(row, state)
            if sig:
                # 전략은 "사유" 또는 (청산비율, "사유") 를 반환할 수 있다
                frac, reason = sig if isinstance(sig, tuple) else (1.0, sig)
                close_trade(row, reason, frac)
            elif scale_in and cash > 0:
                # 추가 매수(물타기) — (보유수량 대비 비율, 사유) 또는 None
                add = scale_in(row, state)
                if add and buy(row, qty_ratio=add[0]) > 0:
                    state["add_count"] = state.get("add_count", 0) + 1
        elif (sig_in := strat.entry(row)):
            # entry() 는 bool 또는 0~1 실수(초기 투입 비중)를 반환할 수 있다
            frac = 1.0 if sig_in is True else float(sig_in)
            state = {"entry_price": row.close, "entry_date": row.Index.strftime("%Y-%m-%d"),
                     "peak": row.high, "entry_reason": strat.entry_reason(row),
                     "last_buy_price": row.close}
            if buy(row, frac_of_cash=frac) <= 0:
                state = {}

        equity.append({"date": row.Index.strftime("%Y-%m-%d"),
                       "equity": round(cash + shares * row.close, 2)})

    if shares > 0:
        close_trade(rows[-1], "기간종료 청산")
        equity[-1]["equity"] = round(cash, 2)

    eq = pd.Series([e["equity"] for e in equity],
                   index=pd.to_datetime([e["date"] for e in equity]))
    mdd = ((eq / eq.cummax()) - 1).min() * 100 if len(eq) else 0.0
    rets = eq.pct_change().dropna()

    ann = np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * ann) if rets.std() > 0 else 0.0

    # --- 추가 위험조정 지표 ---
    # Sortino: 하방 변동성만 벌점 (목표수익률 0 기준 downside deviation)
    downside = rets.clip(upper=0)
    dd_dev = float(np.sqrt((downside ** 2).mean())) if len(rets) else 0.0
    sortino = (rets.mean() / dd_dev * ann) if dd_dev > 0 else 0.0

    # CAGR / Calmar
    years = (eq.index[-1] - eq.index[0]).days / 365.25 if len(eq) > 1 else 0.0
    cagr = ((float(cash) / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    calmar = (cagr / abs(mdd)) if mdd < 0 else 0.0

    # 최장 드로다운 회복 기간(일): 고점 이후 그 고점을 되찾기까지 걸린 최대 일수.
    # 기간 종료까지 회복하지 못했으면 고점~종료일을 세고 ongoing 으로 표시한다.
    peak_val, peak_at = -np.inf, eq.index[0] if len(eq) else None
    longest, ongoing = 0, False
    for ts, v in eq.items():
        if v >= peak_val:
            peak_val, peak_at = v, ts
        else:
            span = (ts - peak_at).days
            if span > longest:
                longest, ongoing = span, False
    if len(eq) and eq.iloc[-1] < peak_val:       # 마지막까지 미회복
        span = (eq.index[-1] - peak_at).days
        if span >= longest:
            longest, ongoing = span, True

    wins = [t for t in trades if t["pnl"] > 0]

    result = {
        "strategy_name": strategy_name, "ticker": ticker, "stop_loss_pct": sl_pct,
        "period": period,
        "start_date": equity[0]["date"], "end_date": equity[-1]["date"],
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": round(float(cash), 2),
        "total_return_pct": round(float(cash) / INITIAL_CAPITAL * 100 - 100, 2),
        "mdd_pct": round(float(mdd), 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "total_trades": len(trades),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(float(sortino), 3),
        "cagr_pct": round(float(cagr), 2),
        "calmar_ratio": round(float(calmar), 3),
        "max_dd_recovery_days": int(longest),
        "dd_recovery_ongoing": int(ongoing),
        "equity_curve_json": json.dumps(equity),
    }
    return result, trades, d
