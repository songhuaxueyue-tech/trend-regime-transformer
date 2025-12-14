"""
label_generator.py

自动生成市场 Regime 标签（Up / Down / Sideway）基于：
 - 窗口线性回归斜率（slope）
 - ATR（波动率）辅助（可选）

主要函数：
 - generate_labels(df, window=48, slope_threshold=1e-4, use_log=True, atr_window=14, volatility_filter=None)
 - recommend_slope_threshold(df, window=48, q=0.90, use_log=True)
 - read_input(path)  # 支持 feather 或 csv
 - CLI: python label_generator.py --in data/BTC.feather --out data/BTC_labeled.feather
"""

import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def read_input(path: str) -> pd.DataFrame:
    """读取 feather 或 csv，返回 DataFrame，要求包含 columns open,high,low,close,volume, 并把时间列设为 index"""
    if path.endswith(".feather") or path.endswith(".parquet"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)
    # 如果有列名 'date' 或 'time'，把它设为 index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    # try to ensure standard column names lower-case
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # Ensure required cols exist
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"input missing required columns: {required - set(df.columns)}")
    # sort by index
    df = df.sort_index()
    return df


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """计算简单的 ATR（平均真实范围）"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def rolling_slope(series: pd.Series, window: int = 48, use_log: bool = True) -> pd.Series:
    """
    用滑动窗口计算线性回归斜率（per bar）。
    如果 use_log=True，则对 close 取对数再回归（对尺度更稳健）。
    返回与原 series 等长的 slope Series（前 window-1 会是 NaN）。
    """
    if use_log:
        series_proc = np.log(series.replace(0, np.nan)).fillna(method='ffill').fillna(method='bfill')
    else:
        series_proc = series

    # numpy.polyfit via pandas rolling.apply
    # 返回 slope per bar (price unit per bar or log-price per bar)
    def _slope(y):
        if len(y) < window:
            return np.nan
        x = np.arange(len(y))
        # fit degree 1 polynomial: slope = coef[0]
        # use np.polyfit which returns highest order first
        try:
            coef = np.polyfit(x, y, 1)
            return float(coef[0])
        except Exception:
            return np.nan

    slope_series = series_proc.rolling(window, min_periods=1).apply(lambda y: _slope(y), raw=True)
    
    return slope_series # 返回滚动窗口斜率


def generate_labels(df: pd.DataFrame,
                    window: int = 48,
                    slope_threshold: Optional[float] = None,
                    slope_quantile_for_threshold: Optional[float] = None,
                    use_log: bool = True,
                    atr_window: int = 14,
                    volatility_filter: Optional[dict] = None,
                    return_probs: bool = False) -> pd.DataFrame:
    """
    生成 regime 标签并返回新的 DataFrame（包含 columns: atr, slope, rel_slope, regime）
    Parameters:
      - window: 窗口长度（bars）
      - slope_threshold: 绝对相对斜率阈值（如果提供则用于判别）
        注意：斜率建议以相对斜率（slope / mean_price）来判断
      - slope_quantile_for_threshold: 如果传入，会根据历史 rel_slope 的分位数自动选择阈值
      - use_log: 是否对 close 做对数再回归
      - atr_window: ATR 计算窗口
      - volatility_filter: 可选 dict 表示如何处理高波动，例如:
            {'atr_multiplier': 2.0, 'use_atr_as_cutoff': True}
      - return_probs: 是否返回 soft-probabilities（占位，暂简单返回 embedding metrics）
    Returns:
      - df_out: 原 df 的拷贝，包含新列 ['atr','slope','rel_slope','regime']
        regime: 0 = sideway, 1 = up, 2 = down
    """
    df = df.copy()
    # compute ATR and slope
    df['atr'] = atr(df, n=atr_window)
    df['slope'] = rolling_slope(df['close'], window=window, use_log=use_log)

    # compute relative slope (slope / mean price over window) to normalize across price scales
    # compute rolling mean price
    rolling_mean = df['close'].rolling(window, min_periods=1).mean()
    df['rel_slope'] = df['slope'] / (rolling_mean.replace(0, np.nan))

    # choose threshold
    if slope_threshold is None:
        if slope_quantile_for_threshold is not None:
            q = slope_quantile_for_threshold
            # choose absolute rel_slope quantile as threshold
            slope_threshold = np.nanquantile(np.abs(df['rel_slope'].dropna()), q)
        else:
            # fallback default - fairly conservative small threshold
            slope_threshold = 1e-4

    # optionally apply volatility filter (if provided, you can tighten/loosen threshold based on ATR)
    # Example volatility_filter: {'atr_multiplier': 1.5, 'use_atr_as_cutoff': False}
    # If use_atr_as_cutoff True, it may label as volatile class (we keep three-class here)
    df['regime'] = 0  # sideway default
    # logic: rel_slope > thresh -> up, < -thresh -> down, else sideway
    df.loc[df['rel_slope'] > slope_threshold, 'regime'] = 1
    df.loc[df['rel_slope'] < -slope_threshold, 'regime'] = 2

    # if volatility_filter provided, optionally mark extremely high-ATR windows as sideway/volatile
    if volatility_filter is not None:
        atr_mult = volatility_filter.get('atr_multiplier', None)
        if atr_mult is not None:
            # compute ATR threshold as quantile or multiplier of median
            if volatility_filter.get('atr_quantile', None) is not None:
                atr_thresh = np.nanquantile(df['atr'].dropna(), volatility_filter['atr_quantile'])
            else:
                atr_thresh = df['atr'].median() * atr_mult
            # if ATR very large, optionally set regime to 0 (sideway / volatile) or flag as 3 (optional)
            if volatility_filter.get('mark_volatile_as', 'sideway') == 'sideway':
                df.loc[df['atr'] > atr_thresh, 'regime'] = 0
            elif volatility_filter.get('mark_volatile_as') == 'volatile_label':
                # mark with 3 (volatile), consumer should handle this class
                df.loc[df['atr'] > atr_thresh, 'regime'] = 3

    if return_probs:
        # placeholder: we do not have probabilistic classifier, but we can return normalized rel_slope as score
        df['regime_score'] = df['rel_slope']  # higher -> up, lower -> down

    return df


def recommend_slope_threshold(df: pd.DataFrame, window: int = 48, q: float = 0.90, use_log: bool = True) -> float:
    """
    根据历史 rel_slope 的分位数给出阈值建议（例如 90% 分位），返回相对斜率阈值
    q: 选择哪个分位（0.90 意味着把前 10% 绝对 rel_slope 认为是趋势）
    """
    tmp = df.copy()
    tmp['slope'] = rolling_slope(tmp['close'], window=window, use_log=use_log)
    rolling_mean = tmp['close'].rolling(window, min_periods=1).mean()
    tmp['rel_slope'] = tmp['slope'] / (rolling_mean.replace(0, np.nan))
    val = np.nanquantile(np.abs(tmp['rel_slope'].dropna()), q)
    return float(val)


def save_df(df: pd.DataFrame, out_path: str):
    if out_path.endswith(".feather") or out_path.endswith(".parquet"):
        df.reset_index().to_feather(out_path)
    else:
        df.to_csv(out_path)


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="input path (.feather or .csv)")
    p.add_argument("--out", dest="outfile", default="labeled.feather", help="output path (.feather or .csv)")
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--slope_threshold", type=float, default=None)
    p.add_argument("--slope_quantile", type=float, default=None,
                   help="if provided, choose threshold as quantile of abs(rel_slope), e.g. 0.90")
    p.add_argument("--use_log", action="store_true", help="use log(close) for regression", default=False)
    p.add_argument("--atr_window", type=int, default=14)
    p.add_argument("--volatility_atr_mult", type=float, default=None,
                   help="if provided, use atr_mult to mark high-volatility windows as sideway")
    return p.parse_args()


def main():
    args = parse_args()
    df = read_input(args.infile)
    use_log = args.use_log
    slope_threshold = args.slope_threshold
    slope_quantile = args.slope_quantile

    if slope_threshold is None and slope_quantile is None:
        # recommend using quantile 0.90 if not provided
        rec = recommend_slope_threshold(df, window=args.window, q=0.90, use_log=use_log)
        print(f"[INFO] No threshold provided. Recommended rel_slope threshold by 90% quantile: {rec:.6e}")
        slope_threshold = rec

    volatility_filter = None
    if args.volatility_atr_mult is not None:
        volatility_filter = {'atr_multiplier': args.volatility_atr_mult, 'mark_volatile_as': 'sideway'}

    labeled = generate_labels(df,
                              window=args.window,
                              slope_threshold=slope_threshold,
                              slope_quantile_for_threshold=None,
                              use_log=use_log,
                              atr_window=args.atr_window,
                              volatility_filter=volatility_filter,
                              return_probs=False)
    # Save: keep the original index as a column for feather
    save_df(labeled, args.outfile)
    print(f"[INFO] Saved labeled data to {args.outfile}")
    print("Regime value counts:\n", labeled['regime'].value_counts(dropna=True))


if __name__ == "__main__":
    main()


