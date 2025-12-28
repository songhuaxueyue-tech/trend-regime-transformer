# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from jinja2.optimizer import optimize
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pygments.lexer import default
from scipy.stats import levene
from technical import qtpylib
from typing import Dict

# This class is a sample. Feel free to customize it.
class CsMom(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = "1h"
    informative_timeframe = "1d"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 杠杆
    lev = 5

    # 风险控制
    risk_pct = DecimalParameter(0.005, 0.03, default = 0.01, space = "buy")

    # 2ema
    fast_ema = IntParameter(10,30,default = 20, space = "buy")
    slow_ema = IntParameter(40,60,default = 50, space = "buy")

    # 3ema
    switch_3ema = BooleanParameter(default = False, space = "buy")
    ema30_timeperiod = IntParameter(10,30,default = 20, space = "buy")
    ema60_timeperiod = IntParameter(40,60,default = 50, space = "buy")
    ema120_timeperiod = IntParameter(70,120,default = 90, space = "buy")

    # adx
    switch_adx = BooleanParameter(default = False, space = "buy")
    adx_timeperiod = IntParameter(7, 35,default = 21, space = "buy")
    adx_min = IntParameter(10,40,default = 20, space = "buy")

    # ema
    switch_ema = BooleanParameter(default = False, space = "buy")
    ema_timeperiod = IntParameter(10,120,default = 90, space = "buy")

    # roi
    roi_n = DecimalParameter(0.1, 10,decimals = 1 ,default = 0.5, space = "sell")

    momentum_win = IntParameter(30, 90, default=60, space="buy", optimize=True)
    vol_win = IntParameter(20, 60, default=30, space="buy", optimize=True)
    top_k = IntParameter(1, 20, default=3, space="buy", optimize=True)

    # atr
    switch_atr_stoploss = BooleanParameter(default=True, space="sell", optimize=False)     # 强制开启atr止损
    atr_timeperiod = IntParameter(7, 35, default = 14, space = "sell") # 新版本atr
    atr_n = IntParameter(1, 10, default = 2, space = "sell")

    # entry_sl
    switch_exit_sl = BooleanParameter(default=False, space="sell",optimize = False)

    # 使用close / current_rate 作为仓位管理标准
    use_close = BooleanParameter(default = True, space = "buy")

    # 缓存：每根主K线刷新
    _rank_long: set = set()
    _rank_short: set = set()
    _sigma_ann: Dict[str, float] = {}

    # 类属性：增加一个缓存（放到类顶部其它缓存旁边）
    _sl_cache: Dict[tuple, float] = {}

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200


    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "fast_ema": {"color":"#F9ED69"},
            "slow_ema": {"color": "#F08A5D"},
            "ema30": {"color": "#F9ED69"},
            "ema60": {"color": "#F08A5D"},
            "ema120": {"color": "#A6E3E9"},
            "ema": {"color": "#E3FDFD"},
            "atr_stoploss_up": {"color": "white"},
            "atr_stoploss_down": {"color": "white"},

        },
        "subplots": {
            "subplots":{
                "adx":{"color": "white"}
            }
        }
    }

    def leverage(self, pair, current_time, current_rate,
                 proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        # 多空同用 20 倍，低流动性币可下调
        return min(self.lev, max_leverage)


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        pairs = self.dp.current_whitelist()
        return [(p, self.informative_timeframe) for p in pairs]


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # 3ema
        if self.switch_3ema.value:
            dataframe["ema30"] = ta.EMA(dataframe, timeperiod = self.ema30_timeperiod.value)
            dataframe["ema60"] = ta.EMA(dataframe, timeperiod = self.ema60_timeperiod.value)
            dataframe["ema120"] = ta.EMA(dataframe, timeperiod = self.ema120_timeperiod.value)
        else:
            # 2ema
            dataframe["fast_ema"] = ta.EMA(dataframe, timeperiod=self.fast_ema.value)
            dataframe["slow_ema"] = ta.EMA(dataframe, timeperiod=self.slow_ema.value)

        if self.switch_adx.value:
            dataframe["adx"] = ta.ADX(dataframe, timeperiod = self.adx_timeperiod.value)

        if self.switch_ema.value:
           dataframe["ema"] = ta.EMA(dataframe, timeperiod = self.ema_timeperiod.value)

        atr = ta.ATR(dataframe, timeperiod = self.atr_timeperiod.value)
        dataframe["atr_stoploss_down"] = dataframe["close"] - atr * self.atr_n.value
        dataframe["atr_stoploss_up"] = dataframe["close"] + atr * self.atr_n.value

        return dataframe

    def bot_loop_start(self, current_time: datetime, ** kwargs):
        """日线横截面动量排名：Top-K 放行入场"""
        pairs = self.dp.current_whitelist()
        L, V = int(self.momentum_win.value), int(self.vol_win.value)

        mom_map, vol_map = {}, {}
        for p in pairs:
            d1 = self.dp.get_pair_dataframe(pair=p, timeframe=self.informative_timeframe)
            if d1 is None or d1.empty or len(d1) < max(L, V) + 2:
                continue

            # L日累计收益（避开前视：取倒数第2根日线）
            ret = d1["close"].pct_change().add(1.0)
            mom_map[p] = float(ret.rolling(L, min_periods=L).apply(np.prod, raw=True).iloc[-2] - 1.0)
            # 年化波动率
            vol_map[p] = float(d1["close"].pct_change().rolling(V, min_periods=V).std().iloc[-2] * np.sqrt(365.0))

        ranked = sorted([p for p in pairs if p in mom_map and p in vol_map],
                        key=lambda q: (mom_map[q], -vol_map[q]), reverse=True)
        self._rank_long = set(ranked[: int(self.top_k.value)])
        self._rank_short = set(ranked[-int(self.top_k.value):])
        self._sigma_ann = vol_map

    def confirm_trade_entry(
            self, pair, order_type, amount, rate, time_in_force,
            current_time, entry_tag, side, **kwargs
    ) -> bool:
        if pair in self._rank_long and side == "long":
            return True

        if pair in self._rank_short and side == "short":
            return True

        return False

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # print(f"populate_entry_trend ")   # 调试

        # 初始化
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        if self.switch_3ema.value:
            enter_long = ((dataframe["ema30"] > dataframe["ema60"]) & (dataframe["ema60"] > dataframe["ema120"]))
            enter_short = ((dataframe["ema30"] < dataframe["ema60"]) & (dataframe["ema60"] < dataframe["ema120"]))
        else:
            enter_long = (dataframe["fast_ema"] > dataframe["slow_ema"])
            enter_short = (dataframe["fast_ema"] < dataframe["slow_ema"])

        if self.switch_adx.value:
            enter_long &= (dataframe["adx"] > self.adx_min.value)
            enter_short &= (dataframe["adx"] > self.adx_min.value)

        if self.switch_ema.value:
            enter_long &= (dataframe["close"] > dataframe["ema"])
            enter_short &= (dataframe["close"] < dataframe["ema"])

        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_short, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        if not self.switch_exit_sl.value:
            return dataframe

        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        exit_long = (dataframe["fast_ema"] < dataframe["slow_ema"])
        exit_short = (dataframe["fast_ema"] > dataframe["slow_ema"])

        dataframe.loc[exit_long,"exit_long"] = 1
        dataframe.loc[exit_short,"exit_short"] = 1

        return dataframe

    use_custom_roi = False

    # 达到固定利率止盈
    def custom_roi(self, pair, trade, current_time, trade_duration, entry_tag, side, **kwargs):
        sigma_ann = self._sigma_ann.get(pair)
        if sigma_ann is None or not np.isfinite(sigma_ann) or sigma_ann <= 0:
            return None  # 回退到 minimal_roi

        # 从年化返回到“日波动”（价格尺度）
        daily_vol = float(sigma_ann / np.sqrt(365.0))

        # 将“价格位移目标”映射为“ROI 阈值”：乘以杠杆
        lev = max(1.0, float(getattr(trade, "leverage", 1) or 1.0))
        roi_tgt = daily_vol * lev * self.roi_n.value

        # 夹在合理区间，避免离谱值（比如极端波动日或超高杠杆）
        return float(roi_tgt)

    # 掉出top 离场
    use_custom_exit = True   # ← 开启自定义退出（与 use_custom_roi 可并存）
    def custom_exit(
            self,
            pair: str,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs,
    ) -> Optional[str]:
        """
        策略性退出：掉出 Top-K/Bottom-K 即离场
        - 多单：不在 self._rank_long 就平仓
        - 空单：不在 self._rank_short 就平仓
        返回字符串标签用于回测统计；返回 None 表示不触发本次退出
        """
        # 启动初期或尚未完成一次排名计算时，避免误触发
        # if not self._rank_long and not self._rank_short:
        #     return None
        #
        # if not trade.is_short:
        #     # 多单掉榜
        #     if pair not in self._rank_long:
        #         return "de-rank"
        # else:
        #     # 空单掉榜（Bottom-K 以 _rank_short 维护）
        #     if pair not in self._rank_short:
        #         return "de-rank"


        sigma_ann = self._sigma_ann.get(pair)
        if sigma_ann is None or not np.isfinite(sigma_ann) or sigma_ann <= 0:
            return None  # 回退到 minimal_roi

        # 从年化返回到“日波动”（价格尺度）
        daily_vol = float(sigma_ann / np.sqrt(365.0))

        # 将“价格位移目标”映射为“ROI 阈值”：乘以杠杆
        lev = max(1.0, float(getattr(trade, "leverage", 1) or 1.0))
        roi_tgt = daily_vol * lev * self.roi_n.value

        if current_profit >= roi_tgt:
            return "take_profit"

        return None

    use_custom_stoploss = True   # ← 必须为 True 才会调用 custom_stoploss
    # ---- 替换原来的 custom_stoploss ----
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if not self.switch_atr_stoploss.value:
            return None

        # 缓存方式获得指定时间之前数据

        # 用 (pair, open_time) 作为 key，避免访问 Trade.custom_data（SQLAlchemy）
        key = (pair, getattr(trade, "open_date_utc", current_time))
        sl_abs = self._sl_cache.get(key)

        if sl_abs is None:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return None

            row = df.iloc[-1]

            atr_down = row.get("atr_stoploss_down")
            atr_up = row.get("atr_stoploss_up")

            if atr_down is None or atr_up is None:
                return None

            sl_abs = float(atr_down if not trade.is_short else atr_up)

            self._sl_cache[key] = sl_abs  # 仅入场第一次计算，之后走缓存


        # # 日期比较方式获得指定时间之前数据
        # df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # if df is None or df.empty:
        #     return None
        
        # df = df[df["date"] < trade.open_date_utc]

        # # print(f"df格式为：{df}")   # 调试

        # row = df.iloc[-1]

        # atr_down = row.get("atr_stoploss_down")
        # atr_up = row.get("atr_stoploss_up")

        # if atr_down is None or atr_up is None:
        #     return None

        # sl_abs = float(atr_down if not trade.is_short else atr_up)


        # 防御：有些回测里 trade.leverage 可能为 None
        lev = float(getattr(trade, "leverage", 1) or 1)

        # print(f"时间为：{current_time}  sl_abs值为：{sl_abs}")  # 调试

        return stoploss_from_absolute(sl_abs, current_rate, is_short=trade.is_short, leverage=lev)

    def custom_stake_amount(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_stake: float,
            min_stake: float | None,
            max_stake: float,
            leverage: float,
            entry_tag: str | None,
            side: str,
            **kwargs
    ) -> float | None:
        """
        目标：若价格打到「初始止损价」，亏损 ≈ 账户本金 * 1%    风险 = stake * stoploss_pct * leverage ≈ equity * target_risk_pct    => stake = equity * target_risk_pct / (stoploss_pct * leverage)    """
        # 账户可用资金（USDT 等）
        free = self.wallets.get_free(self.config["stake_currency"])
        if free is None or free <= 0:
            return None

        risk_stake = free * self.risk_pct.value

        # 风险计算
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None

        last_candle = df.iloc[-1].squeeze()
        if last_candle is None or last_candle.empty:
            return None

        atr_stoploss_higher = last_candle["atr_stoploss_up"]
        atr_stoploss_lower = last_candle["atr_stoploss_down"]

        sl_abs = atr_stoploss_lower if side == "long" else atr_stoploss_higher

        # 自行判断使用close 还是 current_rate
        if self.use_close.value:
            close = last_candle["close"]
            sl_pct = abs(close - sl_abs) / close
        else:
            sl_pct = abs(sl_abs - current_rate) / current_rate

        # 仓位计算
        stake = risk_stake / (sl_pct * max(leverage, 1))

        # 最大仓位限制
        self_max_stake = 100000
        stake = min(stake, self_max_stake / max(leverage, 1))

        return float(stake)


