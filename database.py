"""SQLite 저장/조회 계층."""
import os
import sqlite3
from pathlib import Path

import pandas as pd

# 배포 환경에서 영구 디스크에 DB 를 두려면 DB_PATH 환경변수를 지정한다.
# (예: Render 유료 플랜 디스크 -> /var/data/market_data.db)
DB_PATH = Path(os.getenv("DB_PATH") or (Path(__file__).parent / "market_data.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL, high REAL, low REAL, close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            stop_loss_pct REAL NOT NULL DEFAULT 5.0,
            period        TEXT NOT NULL DEFAULT '3Y',
            start_date    TEXT, end_date TEXT,
            initial_capital REAL,
            final_equity    REAL,
            total_return_pct REAL,
            mdd_pct          REAL,
            win_rate_pct     REAL,
            total_trades     INTEGER,
            sharpe_ratio     REAL,
            sortino_ratio    REAL,
            cagr_pct         REAL,
            calmar_ratio     REAL,
            max_dd_recovery_days INTEGER,
            dd_recovery_ongoing  INTEGER,
            equity_curve_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (strategy_name, ticker, stop_loss_pct, period)
        );

        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            stop_loss_pct REAL NOT NULL DEFAULT 5.0,
            period        TEXT NOT NULL DEFAULT '3Y',
            entry_date TEXT, entry_price REAL,
            exit_date  TEXT, exit_price  REAL,
            shares REAL,
            pnl REAL,
            return_pct REAL,
            entry_reason TEXT,
            exit_reason TEXT,
            holding_days INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_trade_logs
            ON trade_logs(strategy_name, ticker, stop_loss_pct, period);

        -- 파라미터 그리드 서치 결과 캐시
        -- mode='basic'  : tp1/tp2/drop_rate/buy_ratio
        -- mode='detail' : + rsi_buy/sma_period/trailing_stop (숭이 3호 세분화)
        -- 미사용 파라미터는 NULL 이 아니라 0 으로 채운다 (UNIQUE 중복 판정을 위해)
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            mode          TEXT NOT NULL DEFAULT 'basic',
            scope         TEXT NOT NULL,      -- 'ALL' 또는 티커
            period        TEXT NOT NULL,
            stop_loss_pct REAL NOT NULL,
            tp1 REAL DEFAULT 0, tp2 REAL DEFAULT 0,
            drop_rate REAL DEFAULT 0, buy_ratio REAL DEFAULT 0,
            rsi_buy REAL DEFAULT 0, sma_period REAL DEFAULT 0,
            trailing_stop REAL DEFAULT 0,
            total_return_pct REAL, mdd_pct REAL, sharpe_ratio REAL,
            sortino_ratio REAL, cagr_pct REAL, calmar_ratio REAL,
            win_rate_pct REAL, total_trades INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (strategy_name, mode, scope, period, stop_loss_pct,
                    tp1, tp2, drop_rate, buy_ratio,
                    rsi_buy, sma_period, trailing_stop)
        );
        CREATE INDEX IF NOT EXISTS idx_opt
            ON optimization_results(strategy_name, mode, scope, period, stop_loss_pct);

        -- 운영 메타데이터 (마지막 갱신 시각 등)
        CREATE TABLE IF NOT EXISTS app_meta (
            key   TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # optimization_results 에 mode/세부 파라미터 컬럼이 없으면 재생성
        ocols = {r["name"] for r in conn.execute("PRAGMA table_info(optimization_results)")}
        if ocols and not {"mode", "rsi_buy", "sma_period", "trailing_stop"} <= ocols:
            conn.execute("DROP TABLE IF EXISTS optimization_results")
            init_db()
        # 기존 DB 마이그레이션 (entry_reason 컬럼 추가)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(trade_logs)")}
        if "entry_reason" not in cols:
            conn.execute("ALTER TABLE trade_logs ADD COLUMN entry_reason TEXT")

        # 4시간봉 지원 철회 — 관련 테이블 제거 (일봉 전용)
        conn.execute("DROP TABLE IF EXISTS stock_data_4h")

        # stop_loss_pct / period 도입 마이그레이션.
        # backtest_results 는 UNIQUE 제약이 바뀌어 ALTER 로 처리할 수 없으므로
        # 결과 테이블만 재생성한다 (run_backtest.py 재실행으로 복원됨).
        # 신규 지표 컬럼은 UNIQUE 제약과 무관하므로 ALTER 로 덧붙인다
        cur = {r["name"] for r in conn.execute("PRAGMA table_info(backtest_results)")}
        for col, typ in (("sortino_ratio", "REAL"), ("cagr_pct", "REAL"),
                         ("calmar_ratio", "REAL"),
                         ("max_dd_recovery_days", "INTEGER"),
                         ("dd_recovery_ongoing", "INTEGER")):
            if col in cur or "stop_loss_pct" not in cur:
                continue
            conn.execute(f"ALTER TABLE backtest_results ADD COLUMN {col} {typ}")

        # timeframe 컬럼이 남아 있으면 일봉 전용 스키마로 되돌린다
        if not {"stop_loss_pct", "period"} <= cur or "timeframe" in cur:
            conn.executescript("""
                DROP TABLE IF EXISTS backtest_results;
                DROP TABLE IF EXISTS trade_logs;
            """)
            init_db()


def save_prices(ticker: str, df: pd.DataFrame):
    rows = [
        (ticker, idx.strftime("%Y-%m-%d"), float(r.open), float(r.high),
         float(r.low), float(r.close), float(r.volume))
        for idx, r in df.iterrows()
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO stock_data VALUES (?,?,?,?,?,?,?)", rows)


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM stock_data "
            "WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
            conn, params=(ticker, start, end), parse_dates=["date"])
    return df.set_index("date") if not df.empty else df


def purge_unregistered(keep_names):
    """더 이상 등록되지 않은 전략의 결과를 DB 에서 제거한다.
    (전략을 교체하면 옛 레코드가 대시보드에 계속 남기 때문)"""
    keep = list(keep_names)
    q = ",".join("?" * len(keep))
    with get_conn() as conn:
        stale = [r["strategy_name"] for r in conn.execute(
            f"SELECT DISTINCT strategy_name FROM backtest_results "
            f"WHERE strategy_name NOT IN ({q})", keep)]
        if stale:
            conn.execute(f"DELETE FROM backtest_results WHERE strategy_name NOT IN ({q})", keep)
            conn.execute(f"DELETE FROM trade_logs WHERE strategy_name NOT IN ({q})", keep)
    return stale


def set_meta(key: str, value: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO app_meta (key, value, updated_at) VALUES (?,?,datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=datetime('now')", (key, str(value)))


def get_meta(key: str, default=None):
    with get_conn() as conn:
        r = conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def all_meta() -> dict:
    with get_conn() as conn:
        return {r["key"]: r["value"]
                for r in conn.execute("SELECT key, value FROM app_meta")}


def has_results() -> bool:
    """백테스트 결과가 이미 채워져 있는지 (배포 직후 부트스트랩 판단용)."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM backtest_results").fetchone()["c"] > 0


OPT_COLS = ("strategy_name mode scope period stop_loss_pct "
            "tp1 tp2 drop_rate buy_ratio rsi_buy sma_period trailing_stop "
            "total_return_pct mdd_pct sharpe_ratio sortino_ratio cagr_pct "
            "calmar_ratio win_rate_pct total_trades").split()


def load_optimization(strategy, scope, period, sl, mode="basic"):
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM optimization_results WHERE strategy_name=? AND mode=? "
            "AND scope=? AND period=? AND stop_loss_pct=?",
            (strategy, mode, scope, period, sl))]


def save_optimization(rows):
    if not rows:
        return
    cols = ",".join(OPT_COLS)
    ph = ",".join(f":{c}" for c in OPT_COLS)
    with get_conn() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO optimization_results ({cols}) VALUES ({ph})", rows)


def save_backtest(result: dict, trades: list):
    key = (result["strategy_name"], result["ticker"],
           result["stop_loss_pct"], result["period"])
    with get_conn() as conn:
        where = ("WHERE strategy_name=? AND ticker=? AND stop_loss_pct=? AND period=?")
        conn.execute(f"DELETE FROM backtest_results {where}", key)
        conn.execute(f"DELETE FROM trade_logs {where}", key)
        conn.execute("""
            INSERT INTO backtest_results
            (strategy_name, ticker, stop_loss_pct, period, start_date, end_date,
             initial_capital, final_equity, total_return_pct, mdd_pct, win_rate_pct,
             total_trades, sharpe_ratio, sortino_ratio, cagr_pct, calmar_ratio,
             max_dd_recovery_days, dd_recovery_ongoing, equity_curve_json)
            VALUES (:strategy_name,:ticker,:stop_loss_pct,:period,:start_date,
                    :end_date,:initial_capital,:final_equity,:total_return_pct,:mdd_pct,
                    :win_rate_pct,:total_trades,:sharpe_ratio,:sortino_ratio,:cagr_pct,
                    :calmar_ratio,:max_dd_recovery_days,:dd_recovery_ongoing,
                    :equity_curve_json)
        """, result)
        conn.executemany("""
            INSERT INTO trade_logs
            (strategy_name, ticker, stop_loss_pct, period, entry_date,
             entry_price, exit_date, exit_price, shares, pnl, return_pct, entry_reason,
             exit_reason, holding_days)
            VALUES (:strategy_name,:ticker,:stop_loss_pct,:period,:entry_date,
                    :entry_price,:exit_date,:exit_price,:shares,:pnl,:return_pct,
                    :entry_reason,:exit_reason,:holding_days)
        """, trades)
