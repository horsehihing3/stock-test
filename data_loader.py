"""yfinance 일봉 수집 + SQLite 캐싱."""
import pandas as pd
import yfinance as yf

from database import load_prices, save_prices

COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[COLS]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "date"
    # yfinance 가 장중·미확정 봉을 OHLC 없이(NaN) 주는 경우가 있어 제외한다
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_ohlcv(ticker: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
    if not force:
        cached = load_prices(ticker, start, end)
        expected = len(pd.bdate_range(start, end))
        if not cached.empty and len(cached) >= expected * 0.85:
            print(f"  [CACHE] {ticker}: {len(cached)} rows from DB")
            return cached

    print(f"  [FETCH] {ticker}: downloading from yfinance...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"{ticker} 데이터 수집 실패")

    save_prices(ticker, _normalize(df))
    return load_prices(ticker, start, end)
