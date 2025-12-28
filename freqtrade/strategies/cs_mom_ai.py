# --- 1. Imports 放在最顶层 ---
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame 
from datetime import datetime
from typing import Optional, Dict, Union
from freqtrade.strategy import (
    IStrategy, DecimalParameter, IntParameter, BooleanParameter, 
    stoploss_from_absolute, Trade
)
import talib.abstract as ta

# --- 2. AI 路径桥梁 ---
AI_PROJECT_ROOT = Path("/freqtrade/user_data/ai_logic")
if str(AI_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(AI_PROJECT_ROOT))

# --- 3. 尝试导入 ---
try:
    from inference.predictor import RegimePredictor
    print(f"[AI-System] 成功导入预测模块，Docker Path: {AI_PROJECT_ROOT}")
except ImportError as e:
    print(f"[AI-System] 导入失败: {e}")
    RegimePredictor = None


class CsMom(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = True
    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False
    timeframe = "1h"
    informative_timeframe = "1d"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # 杠杆
    lev = 5

    # --- 参数定义 ---
    # 风险控制
    risk_pct = DecimalParameter(0.005, 0.03, default=0.01, space="buy")

    # 2ema
    fast_ema = IntParameter(10, 30, default=20, space="buy")
    slow_ema = IntParameter(40, 60, default=50, space="buy")

    # 3ema
    switch_3ema = BooleanParameter(default=False, space="buy")
    ema30_timeperiod = IntParameter(10, 30, default=20, space="buy")
    ema60_timeperiod = IntParameter(40, 60, default=50, space="buy")
    ema120_timeperiod = IntParameter(70, 120, default=90, space="buy")

    # adx
    switch_adx = BooleanParameter(default=False, space="buy")
    adx_timeperiod = IntParameter(7, 35, default=21, space="buy")
    adx_min = IntParameter(10, 40, default=20, space="buy")

    # ema
    switch_ema = BooleanParameter(default=False, space="buy")
    ema_timeperiod = IntParameter(10, 120, default=90, space="buy")

    # roi
    roi_n = DecimalParameter(0.1, 10, decimals=1, default=0.5, space="sell")

    momentum_win = IntParameter(30, 90, default=60, space="buy", optimize=True)
    vol_win = IntParameter(20, 60, default=30, space="buy", optimize=True)
    top_k = IntParameter(1, 20, default=3, space="buy", optimize=True)

    # atr
    switch_atr_stoploss = BooleanParameter(default=True, space="sell", optimize=False)
    atr_timeperiod = IntParameter(7, 35, default=14, space="sell")
    atr_n = IntParameter(1, 10, default=2, space="sell")

    # entry_sl
    switch_exit_sl = BooleanParameter(default=False, space="sell", optimize=False)

    # 使用 close / current_rate 作为仓位管理标准
    use_close = BooleanParameter(default=True, space="buy")

    # 缓存
    _rank_long: set = set()
    _rank_short: set = set()
    _sigma_ann: Dict[str, float] = {}
    _sl_cache: Dict[tuple, float] = {}

    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Plot config 省略...

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.predictor = None
        if RegimePredictor is not None:
            try:
                self.predictor = RegimePredictor()
                print("[AI-System] 模型加载完毕")
            except Exception as e:
                print(f"[AI-System] 模型初始化失败: {e}")

    def leverage(self, pair, current_time, current_rate,
                 proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        return min(self.lev, max_leverage)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, self.informative_timeframe) for p in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.switch_3ema.value:
            dataframe["ema30"] = ta.EMA(dataframe, timeperiod=self.ema30_timeperiod.value)
            dataframe["ema60"] = ta.EMA(dataframe, timeperiod=self.ema60_timeperiod.value)
            dataframe["ema120"] = ta.EMA(dataframe, timeperiod=self.ema120_timeperiod.value)
        else:
            dataframe["fast_ema"] = ta.EMA(dataframe, timeperiod=self.fast_ema.value)
            dataframe["slow_ema"] = ta.EMA(dataframe, timeperiod=self.slow_ema.value)

        if self.switch_adx.value:
            dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_timeperiod.value)

        if self.switch_ema.value:
            dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_timeperiod.value)

        atr = ta.ATR(dataframe, timeperiod=self.atr_timeperiod.value)
        dataframe["atr_stoploss_down"] = dataframe["close"] - atr * self.atr_n.value
        dataframe["atr_stoploss_up"] = dataframe["close"] + atr * self.atr_n.value

        return dataframe

    def bot_loop_start(self, current_time: datetime, **kwargs):
        pairs = self.dp.current_whitelist()
        L, V = int(self.momentum_win.value), int(self.vol_win.value)

        mom_map, vol_map = {}, {}
        for p in pairs:
            d1 = self.dp.get_pair_dataframe(pair=p, timeframe=self.informative_timeframe)
            if d1 is None or d1.empty or len(d1) < max(L, V) + 2:
                continue
            ret = d1["close"].pct_change().add(1.0)
            mom_map[p] = float(ret.rolling(L, min_periods=L).apply(np.prod, raw=True).iloc[-2] - 1.0)
            vol_map[p] = float(d1["close"].pct_change().rolling(V, min_periods=V).std().iloc[-2] * np.sqrt(365.0))

        ranked = sorted([p for p in pairs if p in mom_map and p in vol_map],
                        key=lambda q: (mom_map[q], -vol_map[q]), reverse=True)
        self._rank_long = set(ranked[: int(self.top_k.value)])
        self._rank_short = set(ranked[-int(self.top_k.value):])
        self._sigma_ann = vol_map

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        # 1. 动量排名检查
        is_ranked_long = (pair in self._rank_long and side == "long")
        is_ranked_short = (pair in self._rank_short and side == "short")
        
        if not (is_ranked_long or is_ranked_short):
            return False

        # 2. AI 检查移交给 custom_stake_amount 处理
        # 这里默认返回 True，因为如果 AI 看反了，我们在仓位里把它砍半甚至砍到 0
        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        if not self.switch_exit_sl.value:
            return dataframe
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        exit_long = (dataframe["fast_ema"] < dataframe["slow_ema"])
        exit_short = (dataframe["fast_ema"] > dataframe["slow_ema"])
        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_short, "exit_short"] = 1
        return dataframe

    def custom_roi(self, pair, trade, current_time, trade_duration, entry_tag, side, **kwargs):
        sigma_ann = self._sigma_ann.get(pair)
        if sigma_ann is None or not np.isfinite(sigma_ann) or sigma_ann <= 0:
            return None
        daily_vol = float(sigma_ann / np.sqrt(365.0))
        lev = max(1.0, float(getattr(trade, "leverage", 1) or 1.0))
        roi_tgt = daily_vol * lev * self.roi_n.value
        return float(roi_tgt)

    use_custom_exit = True
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        sigma_ann = self._sigma_ann.get(pair)
        if sigma_ann is None or not np.isfinite(sigma_ann) or sigma_ann <= 0:
            return None
        daily_vol = float(sigma_ann / np.sqrt(365.0))
        lev = max(1.0, float(getattr(trade, "leverage", 1) or 1.0))
        roi_tgt = daily_vol * lev * self.roi_n.value
        if current_profit >= roi_tgt:
            return "take_profit"
        return None

    use_custom_stoploss = True
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if not self.switch_atr_stoploss.value:
            return None
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
            self._sl_cache[key] = sl_abs
        lev = float(getattr(trade, "leverage", 1) or 1)
        return stoploss_from_absolute(sl_abs, current_rate, is_short=trade.is_short, leverage=lev)


    # -------------------------------------------------------------------------
    # ✅ 融合版仓位管理逻辑：Risk Based + AI Soft Veto (Fixed Scope)
    # -------------------------------------------------------------------------
    use_custom_stake_amount = True   # 开启自定义仓位管理
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: float | None, max_stake: float,
                                leverage: float, entry_tag: str | None, side: str,
                                **kwargs) -> float | None:
            
            # --- 1. 基础风控仓位计算 (保持不变) ---
            free = self.wallets.get_free(self.config["stake_currency"])
            if free is None or free <= 0: return None
            self_max_stake = 100000 
            
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty: return None 

            last_candle = dataframe.iloc[-1].squeeze()
            atr_stoploss_higher = last_candle.get("atr_stoploss_up")
            atr_stoploss_lower = last_candle.get("atr_stoploss_down")
            
            if atr_stoploss_higher is None or atr_stoploss_lower is None:
                return proposed_stake # ATR 还没算出来时，用默认

            sl_abs = atr_stoploss_lower if side == "long" else atr_stoploss_higher

            if self.use_close.value:
                close = last_candle["close"]
                sl_pct = abs(close - sl_abs) / close
            else:
                sl_pct = abs(sl_abs - current_rate) / current_rate

            risk_stake = free * self.risk_pct.value
            if sl_pct == 0: sl_pct = 0.01 # 防止除以0
            
            base_stake = risk_stake / (sl_pct * max(leverage, 1))
            base_stake = min(base_stake, self_max_stake / max(leverage, 1))

            # --- 2. AI 介入逻辑 (关键修复) ---
            ai_factor = 1.0 

            if self.predictor is not None:
                try:
                    # --- [Step 2] 宽松查找 ---
                    # 只要数据日期 <= 信号时间，就认为是可用历史数据
                    valid_rows = dataframe[dataframe['date'] <= current_time]
                    
                    if not valid_rows.empty:
                        current_idx = valid_rows.index[-1]
                        
                        # 确保数据足够长
                        if current_idx < 100: 
                            # 数据太短，无法预测，默认满仓
                            return proposed_stake

                        start_idx = max(0, current_idx - 100)
                        # 注意：iloc 切片是左闭右开，所以要 +1 才能包含 current_idx
                        window_data = dataframe.iloc[start_idx : current_idx + 1]
                        
                        # --- [Step 3] 预测 ---
                        ai_pred = self.predictor.predict(window_data)
                        
                        # --- [Step 4] 决策逻辑 ---
                        is_conflict = False
                        if side == "long" and ai_pred == 1:
                            ai_factor = 0.5
                            is_conflict = True
                            print(f"[AI-Risk] {pair} Long ⚠️ AI Bear (Price Time: {window_data['date'].iloc[-1]}) -> 0.5")
                        elif side == "short" and ai_pred == 0:
                            ai_factor = 0.5
                            is_conflict = True
                            print(f"[AI-Risk] {pair} Short ⚠️ AI Bull (Price Time: {window_data['date'].iloc[-1]}) -> 0.5")
                        
                        # 如果没有冲突，也可以打印一条 Debug 信息证明 AI 运行了 (可选)
                        # else:
                        #    print(f"[AI-Pass] {pair} Consensus. Pred: {ai_pred}")

                    else:
                        print(f"[AI-Warning] No valid data found for {curr_time_naive}. Earliest data: {dataframe['date'].iloc[0] if not dataframe.empty else 'Empty'}")

                except Exception as e:
                    print(f"[AI-Error-Critical] {e}")
                    import traceback
                    traceback.print_exc() # 打印详细堆栈，别让错误跑了
                    ai_factor = 1.0

            # --- 3. 最终计算 ---
            final_stake = base_stake * ai_factor
            if min_stake and final_stake < min_stake: return min_stake
            
            # 调试打印，这行很重要
            if ai_factor != 1.0:
                print(f"[Stake-Debug] Base: {base_stake:.2f}, Factor: {ai_factor}, Final: {final_stake:.2f}")
                
            return float(final_stake)
