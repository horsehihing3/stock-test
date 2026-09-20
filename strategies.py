"""전략 레지스트리: 지표 계산 + 진입/청산 로직."""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 공통 지표


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    upper, lower = mid + mult * std, mid - mult * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """대시보드 주가 차트용 공통 오버레이."""
    out = df.copy()
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma50"] = out["close"].rolling(50).mean()
    _, out["bb_upper"], out["bb_lower"], _ = bollinger(out["close"])
    return out


# ---------------------------------------------------------------- 전략 정의

class Strategy:
    name = ""
    label = ""
    desc = ""
    is_benchmark = False   # True 면 전략 평균/랭킹 집계에서 제외된다

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def entry(self, row) -> bool:
        raise NotImplementedError

    def entry_reason(self, row) -> str:
        """진입 시점의 지표 상태를 사람이 읽을 수 있는 문자열로 기록."""
        return self.label

    def exit(self, row, state: dict):
        """청산 사유(str) 또는 None. 분할청산은 (비율, 사유) 튜플로 반환.
        state: entry_price, entry_date, peak, entry_reason"""
        raise NotImplementedError

    def condition_report(self, row) -> list:
        """진입 조건별 충족 여부 — '오늘의 신호' 화면용.
        [{"label": ..., "ok": bool, "detail": ...}, ...]"""
        return []


class Gemini2(Strategy):
    """단기 추세필터(SMA20) + 완화된 진입조건, 타이트한 손익절.

    [원 사양 수정] 볼린저 중심선 = SMA20 이므로 '%B < 0.35'는 정의상
    '종가 < SMA20'을 의미한다. 따라서 원 사양의 '종가 > SMA20' 필터와
    동시 충족이 불가능하여 거래가 0건이 된다(검증 완료).
    -> 의도(단기 상승추세에서의 눌림목 매수)를 살려 추세 필터를
       'SMA20이 상승 중(5봉 전 대비 상승)'으로 대체한다.
    """
    name, label = "Gemini_2", "제미나이 2호"
    desc = ("SMA20 상승추세 + RSI(14)<=50 & %B<0.35 진입 / "
            "RSI>=65·BB상단·+8%·손절{sl} 청산")
    RSI_BUY, RSI_SELL, PCTB_BUY = 50, 65, 0.35
    SL, TP = -0.04, 0.08
    SLOPE_LOOKBACK = 5

    def indicators(self, df):
        out = df.copy()
        c = out["close"]
        out["sma20"] = c.rolling(20).mean()
        out["sma20_rising"] = out["sma20"] > out["sma20"].shift(self.SLOPE_LOOKBACK)
        out["rsi"] = rsi(c, 14)
        _, out["bb_upper"], out["bb_lower"], out["pct_b"] = bollinger(c, 20, 2.0)
        return out

    def entry(self, row):
        if pd.isna(row.sma20) or pd.isna(row.rsi) or pd.isna(row.pct_b):
            return False
        return (bool(row.sma20_rising) and row.rsi <= self.RSI_BUY
                and row.pct_b < self.PCTB_BUY)

    def entry_reason(self, row):
        return (f"RSI {row.rsi:.1f} & %B {row.pct_b:.2f} "
                f"(SMA20 {row.sma20:.2f} 상승중)")

    def condition_report(self, row):
        """표시 전용 — 진입/청산 판정에는 관여하지 않는다."""
        def num(v, f="{:.2f}"):
            return "-" if pd.isna(v) else f.format(v)
        return [
            {"label": f"SMA20 상승추세 ({self.SLOPE_LOOKBACK}봉 전 대비)",
             "ok": bool(row.sma20_rising),
             "detail": f"SMA20 {num(row.sma20)}"},
            {"label": f"RSI(14) ≤ {self.RSI_BUY}",
             "ok": (not pd.isna(row.rsi)) and row.rsi <= self.RSI_BUY,
             "detail": f"RSI {num(row.rsi, '{:.1f}')}"},
            {"label": f"%B < {self.PCTB_BUY}",
             "ok": (not pd.isna(row.pct_b)) and row.pct_b < self.PCTB_BUY,
             "detail": f"%B {num(row.pct_b)} (BB하단 {num(row.bb_lower)})"},
        ]

    def exit(self, row, state):
        ret = row.close / state["entry_price"] - 1
        if ret <= self.SL:
            return f"손절({self.SL*100:g}%)"
        if ret >= self.TP:
            return "익절(+8%)"
        if row.rsi >= self.RSI_SELL:
            return "RSI>=65"
        if not pd.isna(row.bb_upper) and row.high >= row.bb_upper:
            return "BB 상단터치"
        return None


class Gemini4(Strategy):
    """하이브리드: 대세 상승장에서만 눌림목 진입 + 분할 익절 + 트레일링.

    기존 전략의 약점 보완 의도
      1호 과도한 보수성  -> 진입 조건을 OR(RSI 눌림 | BB 하단터치) 로 완화
      2호 추세 조기 청산 -> +10% 에서 절반만 익절, 나머지는 트레일링으로 계속 보유
      3호 횡보장 손절    -> EMA200 대세 필터로 횡보/하락 국면 진입 자체를 차단
      클로드 1호 큰 MDD  -> 진입 즉시 -5% / -2*ATR 이중 손절로 하단을 조임

    진입: 종가 > EMA200 & (RSI(14) < 45 | 저가가 BB(20,2) 하단 터치)
          & MACD 히스토그램 반등(직전 저점에서 상승 전환)
    청산: [1차] +10% 도달 시 보유 수량의 50% 익절
          [2차] 잔여 50% 는 보유 중 고점 - 2.5*ATR 하락 시 청산
          [손절] 진입가 -5% 또는 진입가 - 2*ATR 도달 시 전량
    """
    name, label = "Gemini_4", "제미나이 4호"
    desc = ("EMA200 대세필터 + (RSI<45 | BB하단터치) + MACD히스토 반등 진입 / "
            "+10% 50% 분할익절 → 잔량 ATR 2.5배 트레일링, 손절{sl}·-2ATR")
    EMA_TREND = 200
    RSI_BUY = 45
    TP1, TP1_FRAC = 0.10, 0.5     # 1차 익절: +10% 에서 50%
    TRAIL_MULT = 2.5              # 2차: 잔량 트레일링
    SL, SL_ATR_MULT = -0.05, 2.0  # 손절: -5% 또는 -2*ATR

    def indicators(self, df):
        out = df.copy()
        c = out["close"]
        out["ema200"] = c.ewm(span=self.EMA_TREND, adjust=False,
                              min_periods=self.EMA_TREND).mean()
        out["rsi"] = rsi(c, 14)
        _, out["bb_upper"], out["bb_lower"], out["pct_b"] = bollinger(c, 20, 2.0)
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        hist = macd - macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = hist
        # 히스토그램이 직전 저점에서 상승 전환(반등)
        out["macd_rebound"] = (hist > hist.shift(1)) & (hist.shift(1) <= hist.shift(2))
        out["atr"] = atr(out, 14)
        out["bb_touch"] = out["low"] <= out["bb_lower"]
        return out

    def entry(self, row):
        if pd.isna(row.ema200) or pd.isna(row.rsi) or pd.isna(row.atr) or pd.isna(row.bb_lower):
            return False
        pullback = row.rsi < self.RSI_BUY or bool(row.bb_touch)
        return row.close > row.ema200 and pullback and bool(row.macd_rebound)

    def entry_reason(self, row):
        trig = []
        if row.rsi < self.RSI_BUY:
            trig.append(f"RSI {row.rsi:.1f}")
        if bool(row.bb_touch):
            trig.append("BB하단터치")
        return (f"{' | '.join(trig)} & MACD히스토 반등 {row.macd_hist:+.2f} "
                f"(종가 {row.close:.2f} > EMA200 {row.ema200:.2f}, ATR {row.atr:.2f})")

    def condition_report(self, row):
        def num(v, f="{:.2f}"):
            return "-" if pd.isna(v) else f.format(v)
        trend = not pd.isna(row.ema200) and row.close > row.ema200
        rsi_ok = not pd.isna(row.rsi) and row.rsi < self.RSI_BUY
        bb_ok = bool(row.bb_touch)
        return [
            {"label": "대세 상승 (종가 > EMA200)", "ok": bool(trend),
             "detail": f"종가 {num(row.close)} / EMA200 {num(row.ema200)}"},
            {"label": f"눌림목 (RSI<{self.RSI_BUY} 또는 BB하단 터치)", "ok": rsi_ok or bb_ok,
             "detail": f"RSI {num(row.rsi, '{:.1f}')} / BB하단 {num(row.bb_lower)}"
                       + (" · 터치" if bb_ok else "")},
            {"label": "MACD 히스토그램 반등", "ok": bool(row.macd_rebound),
             "detail": f"히스토 {num(row.macd_hist, '{:+.2f}')}"},
        ]

    def exit(self, row, state):
        entry_px = state["entry_price"]
        ret = row.close / entry_px - 1
        # 손절: -5% 또는 진입가 - 2*ATR (둘 중 먼저 닿는 쪽)
        atr_stop = entry_px - self.SL_ATR_MULT * row.atr if not pd.isna(row.atr) else None
        if ret <= self.SL:
            return f"손절({self.SL*100:g}%)"
        if atr_stop is not None and row.close <= atr_stop:
            return f"손절(-{self.SL_ATR_MULT}ATR)"
        # 1차 익절: +10% 에서 50%
        if not state.get("scaled") and ret >= self.TP1:
            state["scaled"] = True
            return (self.TP1_FRAC, f"1차 익절(+10%, 수량 {int(self.TP1_FRAC*100)}%)")
        # 2차: 잔여 수량 트레일링
        if state.get("scaled") and not pd.isna(row.atr) \
                and row.close <= state["peak"] - self.TRAIL_MULT * row.atr:
            return f"2차 트레일링({self.TRAIL_MULT}ATR)"
        return None


class Claude2(Strategy):
    """변동성 예산 기반 조기 재진입 (Volatility-Budgeted Early Re-entry).

    설계 문서: DESIGN_Claude_2.md

    문제의식 — 최장 드로다운 회복일은 '가격'이 아니라 '자산곡선' 위에서 측정된다.
    현금 구간에서 자산곡선은 평평하고, 평평한 선은 고점을 회복하지 못한다.
    기존 5개 전략이 20개 조합 중 9개에서 미회복인 이유가 이것이다.
    회복기간을 줄이는 유일한 방법은 회복 구간에 투자되어 있는 것이다.

    진입: 이동평균을 전혀 보지 않는다. 기존 5개는 모두 장기 MA 위치를 게이트로
          쓰기 때문에 바닥에서 57~123일 뒤에야 재진입한다. 여기서는
            (a) 변동성 수축  — ATR% 의 252봉 백분위 < 0.5
            (b) 국소 저점 대비 ATR 단위 반등 — close - min(close,20) >= need
          만 본다. 실측상 바닥 +8~22일에 복귀한다.

    과최적화 방지 — 티커별 상수·분기문을 쓰지 않는다. 종목 차이는 자산 자신의
    변동성 백분위(vol_rank)와 낙폭(dd)에서 자동으로 나온다.
      need    = BASE_K * (1 + KDD * dd) * ATR   낙폭이 깊을수록 진입 문턱 상승
      k_trail = TRAIL_HI - TRAIL_LO * vol_rank  변동성이 높을수록 트레일 계수 축소

    청산 — 방어선이 서로를 죽이지 않도록 max() 로 결합한다.
      stop = max(entry*(1+SL), peak - k_trail*ATR)
      포지션 초반·고변동에서는 SL(손절 스윕값)이, 이익 구간에서는 트레일링이
      binding 한다. 덕분에 손절 스윕 3~30% 가 죽은 파라미터가 되지 않는다.
    """
    name, label = "Claude_2", "클로드 2호"
    desc = ("변동성 수축 + 국소저점 대비 ATR 반등 진입(MA 미사용, 낙폭비례 문턱) / "
            "수습손절·고점-1ATR 50% 축소·max(손절{sl}, ATR트레일링) 청산")

    # 파라미터는 4종목 x 2구간 = 8셀 그리드 스윕(81조합)에서
    # '전 셀 합격 수 -> Calmar 중앙값 -> 최솟값' 순으로 선정했다 (사전 고정 기준).
    # 최고점이 아니라 일관성 기준이며, 이웃(+-1스텝) 안정성도 확인했다.
    #   TIER_ATR=1.0 은 구조적 선택 (8/8 조합 4개 전부가 1.0, 합격셀 평균 6.1/8)
    #   KDD=3 은 전멸 (8/8 조합 0개) — 낙폭 비례 문턱이 약하면 하락장에서 무너진다
    BASE_K = 0.8        # 기본 요구 반등폭 (ATR 배수)
    KDD = 6.0           # 낙폭 비례 계수
    VR = 0.5            # 변동성 백분위 상한
    TRAIL_HI = 3.0      # k_trail 상한
    TRAIL_LO = 1.2      # vol_rank 1 일 때 차감폭
    PROB_K = 1.0        # 수습 기간 손절 (ATR 배수)
    PROB_BARS = 10      # 수습 기간 길이 (봉)
    TIER_ATR = 1.0      # 고점 대비 이 ATR 만큼 밀리면
    TIER_FRAC = 0.50    # 보유 수량의 이 비율을 축소
    COOLDOWN = 10       # 신호 디바운스 (직전 신호 이후 최소 간격, 봉)
    SL = -0.05          # 손절 스윕이 덮어쓴다

    def indicators(self, df):
        out = df.copy()
        c = out["close"]
        a = atr(out, 14)
        out["atr"] = a
        atr_pct = a / c
        # 자기 자신의 과거와만 비교 — 절대 가격·티커 무관
        out["vol_rank"] = atr_pct.rolling(252, min_periods=252).rank(pct=True)
        out["dd"] = 1 - c / c.rolling(252, min_periods=60).max()
        out["low20"] = c.rolling(20).min()
        out["need"] = self.BASE_K * (1 + self.KDD * out["dd"]) * a
        out["k_trail"] = self.TRAIL_HI - self.TRAIL_LO * out["vol_rank"]

        sig = ((c - out["low20"]) >= out["need"]) & (out["vol_rank"] < self.VR)
        if self.COOLDOWN > 0:
            # 신호 디바운스: 직전 신호로부터 COOLDOWN 봉이 지나야 다시 유효.
            # 과거 인덱스만 참조하므로 미래 정보 유입 없음.
            keep, last = [], -10 ** 9
            for i, s in enumerate(sig.to_numpy()):
                ok = bool(s) and (i - last) >= self.COOLDOWN
                if ok:
                    last = i
                keep.append(ok)
            sig = pd.Series(keep, index=out.index)
        out["sig"] = sig
        return out

    def entry(self, row):
        return bool(row.sig)

    def entry_reason(self, row):
        lift = (row.close - row.low20) / row.atr
        return (f"저점대비 {lift:.2f}ATR ≥ 요구 {row.need / row.atr:.2f}ATR "
                f"(낙폭 {row.dd * 100:.0f}%, 변동성분위 {row.vol_rank:.2f}, ATR {row.atr:.2f})")

    def condition_report(self, row):
        def num(v, f="{:.2f}"):
            return "-" if pd.isna(v) else f.format(v)
        lift_ok = (not pd.isna(row.need)) and (row.close - row.low20) >= row.need
        vol_ok = (not pd.isna(row.vol_rank)) and row.vol_rank < self.VR
        return [
            {"label": "변동성 수축 (ATR% 252봉 분위 < 0.5)", "ok": bool(vol_ok),
             "detail": f"분위 {num(row.vol_rank)}"},
            {"label": "국소저점 대비 반등 ≥ 요구폭", "ok": bool(lift_ok),
             "detail": f"{num((row.close - row.low20) / row.atr)}ATR / "
                       f"요구 {num(row.need / row.atr)}ATR (낙폭 {num(row.dd * 100, '{:.0f}')}%)"},
            {"label": "신호 디바운스 통과", "ok": bool(row.sig),
             "detail": f"최소 간격 {self.COOLDOWN}봉"},
        ]

    def exit(self, row, state):
        a = row.atr
        if pd.isna(a) or a <= 0:
            return None
        state["bars"] = state.get("bars", 0) + 1
        gain_atr = (row.close - state["entry_price"]) / a
        state["max_gain_atr"] = max(state.get("max_gain_atr", -9e9), gain_atr)

        # 수습 기간: 포지션이 스스로를 증명(+1ATR)하기 전까지는 좁게 자른다
        if state["bars"] <= self.PROB_BARS and state["max_gain_atr"] < 1.0 \
                and row.close <= state["entry_price"] - self.PROB_K * a:
            return f"수습 손절({self.PROB_K}ATR)"

        floor_px = state["entry_price"] * (1 + self.SL)
        trail_px = state["peak"] - row.k_trail * a
        if row.close <= max(floor_px, trail_px):
            return f"손절({self.SL * 100:g}%)" if floor_px >= trail_px \
                else f"트레일링({row.k_trail:.1f}ATR)"

        # 단계 축소: 고점 대비 1ATR 밀리면 절반을 덜어 노출을 줄인다
        if not state.get("tier") and (state["peak"] - row.close) / a >= self.TIER_ATR:
            state["tier"] = 1
            return (self.TIER_FRAC, f"위험 축소(고점-{self.TIER_ATR}ATR, "
                                    f"{int(self.TIER_FRAC * 100)}%)")
        return None


class SoongI1(Strategy):
    """숭이전략 — 상시 보유 + 물타기(분할 매수) + 평단 기준 분할 익절.

    지표 신호를 쓰지 않는다. 포지션이 없으면 바로 진입하고, 떨어지면 사서
    평단을 낮추고, 평단 대비 일정 수익이 나면 나눠서 판다.

    진입: 무포지션이면 즉시 매수. 초기 투입은 자본의 INIT_FRAC 만 쓰고
          나머지는 추가 매수용 현금으로 남긴다.
          (원 사양에 초기 비중이 없어 50% 로 잡았다. 100% 를 넣으면 현금이
           없어 물타기 자체가 불가능하다 — 아래 '한계' 참조)
    추가 매수: 직전 매수가 대비 -DROP_STEP 하락할 때마다
          현재 보유 수량의 ADD_RATIO 만큼 추가 매수 → 평단 하향
    익절: 평단 대비 +TP2 -> 잔여 전량 청산(포지션 리셋)
          평단 대비 +TP1 -> 보유 수량의 50% 익절 (1회만)

    한계(실측 필요): 손절이 없다. 손절 스윕 파라미터가 한 번도 발동하지 않아
    8단계 결과가 전부 동일해진다. 하락장에서는 현금이 소진될 때까지 물타다가
    그대로 버티는 구조다.
    """
    name, label = "SoongI_1", "숭이전략"
    desc = ("무조건 진입 + 직전 매수가 -10% 마다 보유수량 20% 물타기 / "
            "평단 +10% 절반 익절 → 평단 +30% 전량 청산 (손절 없음)")

    INIT_FRAC = 0.50     # 최초 진입에 쓰는 현금 비중
    DROP_STEP = 0.10     # 직전 매수가 대비 하락률
    ADD_RATIO = 0.20     # 추가 매수 수량 = 보유 수량 x 이 비율
    TP1, TP1_FRAC = 0.10, 0.50
    TP2 = 0.30
    SL = -1.0            # 미사용 (손절 로직 없음)

    def indicators(self, df):
        return df.copy()

    def entry(self, row):
        return self.INIT_FRAC          # 0~1 실수 = 초기 투입 비중

    def entry_reason(self, row):
        return (f"숭이 최초 진입 (자본 {self.INIT_FRAC:.0%} 투입, "
                f"종가 {row.close:.2f})")

    def scale_in(self, row, state):
        """직전 매수가 대비 -10% 마다 보유 수량의 20% 추가 매수."""
        last = state.get("last_buy_price")
        if last and row.close <= last * (1 - self.DROP_STEP):
            return (self.ADD_RATIO,
                    f"물타기(직전 매수가 {last:.2f} 대비 "
                    f"{(row.close / last - 1) * 100:.1f}%)")
        return None

    def exit(self, row, state):
        ret = row.close / state["entry_price"] - 1     # entry_price = 평단가
        if ret >= self.TP2:
            return f"2차 완익절(평단 +{self.TP2 * 100:g}%)"
        if not state.get("scaled") and ret >= self.TP1:
            state["scaled"] = True
            return (self.TP1_FRAC,
                    f"1차 반익절(평단 +{self.TP1 * 100:g}%, 수량 50%)")
        return None

    def condition_report(self, row):
        return [{"label": "상시 진입 (무포지션이면 즉시 매수)", "ok": True,
                 "detail": f"초기 투입 {self.INIT_FRAC:.0%}, "
                           f"-{self.DROP_STEP * 100:g}% 마다 물타기"}]


class SoongI2(Strategy):
    """숭이2 전략 — 숭이전략에 진입 타이밍·물타기 상한·트레일링·비상손절 추가.

    숭이전략(SoongI_1)의 약점 보완
      무조건 진입 -> 눌림목에서만 진입 (고점 매수 완화)
      무한 물타기 -> 총 매수 4회 상한 (하락장 자금 고갈 방지)
      +30% 고정 익절 -> 1차 익절 후 트레일링으로 추세를 더 태움
      손절 없음    -> 물타기 소진 후 평단 -15% 비상 손절 (스윕 파라미터 연동)

    진입: 무포지션 & (RSI(14) <= 50 또는 종가 <= SMA20) 일 때 자본의
          INIT_FRAC 만 투입. 나머지는 물타기용 현금으로 남긴다.
    추가 매수: 직전 매수가 대비 -10% 마다 보유 수량의 20% 추가.
          단 초기 매수 포함 총 MAX_BUYS 회까지만.
    익절: [1차] 평단 +10% -> 50% 익절 (1회)
          [2차] 1차 익절 후, 평단 +20% 도달 또는 보유 중 고점 대비 -5%
                -> 잔여 전량 청산 후 포지션 리셋
    손절: 매수를 MAX_BUYS 회 모두 소진한 뒤 평단 대비 SL 하락 시 전량
          (SL 기본 -15%, 손절 스윕이 덮어쓴다)
    """
    name, label = "SoongI_2", "숭이2 전략"
    desc = ("눌림목(RSI≤50 | 종가≤SMA20) 진입 + -10%마다 20% 물타기(총 4회 한도) / "
            "평단 +10% 절반익절 → +20%·고점-5% 트레일링 청산, 물타기 소진 후 비상손절{sl}")

    INIT_FRAC = 0.40      # 최초 진입 비중 (4회 매수를 커버하도록 1호보다 낮춤)
    DROP_STEP = 0.10      # 직전 매수가 대비 하락률
    ADD_RATIO = 0.20      # 추가 매수 수량 = 보유 수량 x 이 비율
    MAX_BUYS = 4          # 초기 매수 포함 총 매수 횟수 상한
    RSI_BUY, SMA_LEN = 50, 20
    TP1, TP1_FRAC = 0.10, 0.50
    TP2 = 0.20            # 1차 익절 후 완익절 목표
    TRAIL = 0.05          # 1차 익절 후 고점 대비 허용 하락폭
    SL = -0.15            # 비상 손절 (손절 스윕이 덮어쓴다)

    def indicators(self, df):
        out = df.copy()
        c = out["close"]
        out["sma20"] = c.rolling(self.SMA_LEN).mean()
        out["rsi"] = rsi(c, 14)
        return out

    def _pullback(self, row):
        rsi_ok = (not pd.isna(row.rsi)) and row.rsi <= self.RSI_BUY
        sma_ok = (not pd.isna(row.sma20)) and row.close <= row.sma20
        return rsi_ok, sma_ok

    def entry(self, row):
        rsi_ok, sma_ok = self._pullback(row)
        return self.INIT_FRAC if (rsi_ok or sma_ok) else False

    def entry_reason(self, row):
        rsi_ok, sma_ok = self._pullback(row)
        why = []
        if rsi_ok:
            why.append(f"RSI {row.rsi:.1f}≤{self.RSI_BUY}")
        if sma_ok:
            why.append(f"종가 {row.close:.2f}≤SMA20 {row.sma20:.2f}")
        return f"눌림목 진입 ({' | '.join(why)}, 자본 {self.INIT_FRAC:.0%})"

    def condition_report(self, row):
        def num(v, f="{:.2f}"):
            return "-" if pd.isna(v) else f.format(v)
        rsi_ok, sma_ok = self._pullback(row)
        return [
            {"label": f"RSI(14) ≤ {self.RSI_BUY}", "ok": rsi_ok,
             "detail": f"RSI {num(row.rsi, '{:.1f}')}"},
            {"label": "종가 ≤ SMA20", "ok": sma_ok,
             "detail": f"종가 {num(row.close)} / SMA20 {num(row.sma20)}"},
            {"label": "진입 = 둘 중 하나만 충족", "ok": rsi_ok or sma_ok,
             "detail": f"초기 {self.INIT_FRAC:.0%} 투입 후 최대 {self.MAX_BUYS}회 분할"},
        ]

    def scale_in(self, row, state):
        """직전 매수가 -10% 마다 20% 추가. 총 매수 MAX_BUYS 회로 제한."""
        if 1 + state.get("add_count", 0) >= self.MAX_BUYS:
            return None
        last = state.get("last_buy_price")
        if last and row.close <= last * (1 - self.DROP_STEP):
            return (self.ADD_RATIO,
                    f"물타기 {state.get('add_count', 0) + 1}/{self.MAX_BUYS - 1}"
                    f"(직전 {last:.2f} 대비 {(row.close / last - 1) * 100:.1f}%)")
        return None

    def exit(self, row, state):
        ret = row.close / state["entry_price"] - 1          # entry_price = 평단
        buys = 1 + state.get("add_count", 0)

        # 비상 손절 — 물타기를 모두 소진한 뒤에만 발동
        if buys >= self.MAX_BUYS and ret <= self.SL:
            return f"비상 손절(평단 {self.SL * 100:g}%, 매수 {buys}회 소진)"

        if state.get("scaled"):
            # 1차 익절 후: 목표 도달 또는 고점 되돌림
            if ret >= self.TP2:
                return f"2차 완익절(평단 +{self.TP2 * 100:g}%)"
            if row.close <= state["peak"] * (1 - self.TRAIL):
                return f"2차 트레일링(고점 대비 -{self.TRAIL * 100:g}%)"
            return None

        if ret >= self.TP1:
            state["scaled"] = True
            state["peak"] = max(state["peak"], row.high)
            return (self.TP1_FRAC,
                    f"1차 반익절(평단 +{self.TP1 * 100:g}%, 수량 50%)")
        return None


class SoongI3(Strategy):
    """숭이 3호 — 숭이2의 '조기 청산' 문제를 고친 버전.

    숭이2 실측 진단 (3Y, 손절 5%)
      · 2차 완익절 손익 $28,753 -> $9,295 로 $19,458 증발
      · 고점 -5% 트레일링이 43건 발동, 보유 기간 224일 -> 95일로 단축
      · 비상 손절은 3건/-$6,556 로 영향이 작았고, 진입 기회는 오히려 2배 증가
      => 문제는 손절도 진입도 아니고 '너무 빨리 나가는 것'이었다.

    3호의 수정점
      · 트레일링을 '평단 +15% 도달 이후'에만 가동 (그 전에는 흔들려도 버틴다)
      · 트레일링 폭을 고점 -5% -> -12% 로 확대
      · 완익절 목표를 +20% -> +30% 로 되돌림 (숭이1 수준)
      · 진입 조건을 RSI<=55 단일로 완화 (숭이2의 RSI<=50 | 종가<=SMA20 보다 단순)

    진입: 무포지션 & RSI(14) <= 55 -> 자본의 INIT_FRAC 투입
    추가 매수: 직전 매수가 -10% 마다 보유 수량 50%, 총 MAX_BUYS 회 한도
    익절: 평단 +60% 전량 / 평단 +10% 에서 50% (1회)
    트레일링: 평단 +15% 를 한 번이라도 찍은 뒤, 고점 대비 -12% 시 잔여 전량
    손절: 매수 MAX_BUYS 회 소진 후 평단 대비 SL(기본 -15%) 하락 시 전량

    [1차 그리드 서치] 752조합(tp1/tp2/물타기/매수량) 결과 TP2 0.30 -> 0.60,
    ADD_RATIO 0.20 -> 0.50 채택. 3Y Calmar 1.27 -> 1.51.

    [2차 세분화 그리드] 3072조합(RSI/SMA/매수량/물타기/tp1/트레일링) 결과
    RSI_BUY 55 -> 45, SMA_PERIOD 0 -> 20, ADD_RATIO 0.50 -> 0.40 채택.
      3Y  Calmar 1.51 -> 1.57 · MDD -11.37% -> -10.73% · 승률 97.0%
    DROP_STEP(0.10) / TP1(0.10) / TRAIL(0.12) 은 세분화 1위와 동일해 유지.

    주의: DROP_STEP·TRAIL 은 '크기'로 저장한다(부호 없음).
          코드에서 close <= last * (1 - DROP_STEP) 형태로 쓰이므로
          음수를 넣으면 조건이 반대로 뒤집힌다.
    """
    name, label = "SoongI_3", "숭이 3호"
    desc = ("(RSI≤45 | 종가≤SMA20) 진입 + -10%마다 40% 물타기(총 4회) / "
            "평단 +10% 절반익절 · +60% 완익절 · +15% 도달 후 고점-12% 트레일링 · "
            "4회 소진 후 비상손절{sl}")

    INIT_FRAC = 0.40      # 최초 진입에 투입하는 자본 비중
    DROP_STEP = 0.10      # 물타기 간격 (-10%)
    ADD_RATIO = 0.40      # 1회 추가매수 수량 (세분화 최적: 0.50 -> 0.40)
    MAX_BUYS = 4
    RSI_BUY = 45          # 세분화 최적: 55 -> 45
    SMA_PERIOD = 20       # 세분화 최적: 미사용 -> SMA20 OR 진입 경로 추가
    TP1, TP1_FRAC = 0.10, 0.50
    TP2 = 0.60            # 완익절
    TRAIL_ARM = 0.15      # 이 수익률을 찍은 뒤에만 트레일링 가동
    TRAIL = 0.12          # 고점 대비 허용 하락폭 (-12%)
    SL = -0.15            # 비상 손절 (손절 스윕이 덮어쓴다)

    # SMA_PERIOD > 0 이면 '종가 <= SMA(n)' 를 OR 진입 경로로 추가한다 (0 = RSI 단독).
    def indicators(self, df):
        out = df.copy()
        out["rsi"] = rsi(out["close"], 14)
        out["sma_n"] = (out["close"].rolling(self.SMA_PERIOD).mean()
                        if self.SMA_PERIOD else float("nan"))
        return out

    def _paths(self, row):
        rsi_ok = (not pd.isna(row.rsi)) and row.rsi <= self.RSI_BUY
        sma_ok = bool(self.SMA_PERIOD) and (not pd.isna(row.sma_n)) \
            and row.close <= row.sma_n
        return rsi_ok, sma_ok

    def entry(self, row):
        rsi_ok, sma_ok = self._paths(row)
        return self.INIT_FRAC if (rsi_ok or sma_ok) else False

    def entry_reason(self, row):
        rsi_ok, sma_ok = self._paths(row)
        why = []
        if rsi_ok:
            why.append(f"RSI {row.rsi:.1f} ≤ {self.RSI_BUY}")
        if sma_ok:
            why.append(f"종가 {row.close:.2f} ≤ SMA{self.SMA_PERIOD:g} {row.sma_n:.2f}")
        return (f"{' | '.join(why)} 진입 "
                f"(자본 {self.INIT_FRAC:.0%}, 종가 {row.close:.2f})")

    def condition_report(self, row):
        rsi_ok, sma_ok = self._paths(row)
        out = [{"label": f"RSI(14) ≤ {self.RSI_BUY}", "ok": rsi_ok,
                "detail": ("-" if pd.isna(row.rsi) else f"RSI {row.rsi:.1f}")
                          + f" · 초기 {self.INIT_FRAC:.0%} 투입, "
                            f"최대 {self.MAX_BUYS}회 분할"}]
        if self.SMA_PERIOD:
            out.append({"label": f"종가 ≤ SMA{self.SMA_PERIOD:g}", "ok": sma_ok,
                        "detail": ("-" if pd.isna(row.sma_n)
                                   else f"SMA {row.sma_n:.2f}")})
        return out

    def scale_in(self, row, state):
        if 1 + state.get("add_count", 0) >= self.MAX_BUYS:
            return None
        last = state.get("last_buy_price")
        if last and row.close <= last * (1 - self.DROP_STEP):
            return (self.ADD_RATIO,
                    f"물타기 {state.get('add_count', 0) + 1}/{self.MAX_BUYS - 1}"
                    f"(직전 {last:.2f} 대비 {(row.close / last - 1) * 100:.1f}%)")
        return None

    def exit(self, row, state):
        ret = row.close / state["entry_price"] - 1
        buys = 1 + state.get("add_count", 0)

        if buys >= self.MAX_BUYS and ret <= self.SL:
            return f"비상 손절(평단 {self.SL * 100:g}%, 매수 {buys}회 소진)"

        if ret >= self.TP2:
            return f"2차 완익절(평단 +{self.TP2 * 100:g}%)"

        if not state.get("scaled") and ret >= self.TP1:
            state["scaled"] = True
            return (self.TP1_FRAC,
                    f"1차 반익절(평단 +{self.TP1 * 100:g}%, 수량 50%)")

        # 트레일링은 '+15% 를 한 번이라도 찍은 뒤'에만 가동
        if ret >= self.TRAIL_ARM:
            state["armed"] = True
        if state.get("armed") and row.close <= state["peak"] * (1 - self.TRAIL):
            return f"트레일링(고점 대비 -{self.TRAIL * 100:g}%)"
        return None


class BuyHold(Strategy):
    """벤치마크: 첫 봉 종가 전량 매수 후 마지막 봉까지 보유.

    별도 계산 경로를 만들지 않고 동일한 백테스트 엔진을 그대로 통과시키기 위해
    Strategy 인터페이스로 구현한다.
      - entry: 항상 True. 엔진은 무포지션일 때만 entry 를 검사하므로
               백테스트 구간의 첫 봉에서 한 번 진입하고 이후로는 호출되지 않는다.
               (지표 워밍업 구간은 엔진이 잘라낸 뒤라 '첫 봉'을 전략이 직접 알 수 없다)
      - exit : 항상 None -> 엔진이 마지막 봉에서 '기간종료 청산' 처리
    따라서 수수료(진입/청산 각 0.1%)·MDD·Sharpe 계산이 전략들과 완전히 동일하다.
    손절 파라미터는 청산 로직이 없으므로 영향을 주지 않는다(전 손절 레벨 동일 결과).
    """
    name, label = "Buy_Hold", "Buy & Hold"
    desc = "벤치마크 — 기간 첫 봉 종가 전량 매수 후 마지막 봉까지 보유 (수수료 왕복 0.2%)"
    is_benchmark = True
    SL = -1.0   # 사실상 미사용 (exit 이 항상 None)

    def indicators(self, df):
        return df.copy()

    def entry(self, row):
        return True

    def entry_reason(self, row):
        return f"벤치마크 매수 (첫 봉 종가 {row.close:.2f})"

    def exit(self, row, state):
        return None   # 청산 신호 없음 -> 기간 종료 시 엔진이 청산

    def condition_report(self, row):
        return [{"label": "기간 내 상시 보유", "ok": True, "detail": "청산 조건 없음"}]


CLASSES = (SoongI3, Gemini2, Claude2, Gemini4, SoongI1, SoongI2, BuyHold)
REGISTRY = {c.name: c() for c in CLASSES}          # 메타데이터용 기본 인스턴스
LABELS = {n: s.label for n, s in REGISTRY.items()}
BENCHMARKS = [n for n, s in REGISTRY.items() if s.is_benchmark]
TRADING = [n for n, s in REGISTRY.items() if not s.is_benchmark]

DEFAULT_STOP_LOSS_PCT = 5.0
# 손절 스윕 범위.
# 하한을 3% -> 1% 로 내렸다: 저변동 자산의 ATR% 중앙값이 AAPL 2.18%, QQQ 1.58% 라
# 하한 3% 로는 ATR 기반 손절이 항상 먼저 걸려 SL 이 구조적으로 도달 불가능했다.
STOP_LOSS_LEVELS = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0)


def get(name: str, stop_loss_pct: float | None = None) -> Strategy:
    """stop_loss_pct 를 주면 손절선만 덮어쓴 새 인스턴스를 반환한다.

    (전략 인스턴스는 공유되므로 반드시 새 객체를 만들어 동시 요청 간 간섭을 막는다.)
    """
    if stop_loss_pct is None:
        return REGISTRY[name]
    s = {c.name: c for c in CLASSES}[name]()
    s.SL = -abs(float(stop_loss_pct)) / 100.0
    s.desc = _render_desc(type(s).desc, stop_loss_pct)
    return s


def _render_desc(desc: str, pct: float) -> str:
    """desc 의 {sl} 자리표시자에만 실제 손절값을 넣는다.

    (이전 구현은 desc 안의 모든 '-숫자%' 를 치환해서 물타기 -10%, 트레일링
     -12% 같은 무관한 수치까지 손절값으로 바꿔버렸다.)
    """
    return desc.replace("{sl}", f" -{abs(pct):g}%")


# 기본 인스턴스의 desc 도 자기 SL 로 렌더링해 둔다
for _s in REGISTRY.values():
    _s.desc = _render_desc(type(_s).desc, abs(_s.SL) * 100)
