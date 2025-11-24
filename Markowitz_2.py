"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 資產（不含 SPY）
        assets = self.price.columns[self.price.columns != self.exclude]

        # 參數：可以之後再微調
        lookback_short = 63    # 3 個月動能
        lookback_long = 252    # 12 個月動能
        vol_lookback   = 20    # 1 個月波動
        trend_ma       = 200   # SPY 趨勢線
        top_k          = 3     # 挑前 3 強

        # 初始化權重，全 0
        self.portfolio_weights = pd.DataFrame(
            0.0, index=self.price.index, columns=self.price.columns
        )

        spy_price = self.price[self.exclude]
        spy_ma = spy_price.rolling(trend_ma).mean()

        # 至少要有 long lookback & MA 才能開始
        start_idx = max(lookback_long, trend_ma, vol_lookback)

        for i in range(start_idx, len(self.price)):
            date = self.price.index[i]

            # ===== 1. 市場 regime：用 SPY 看多空 =====
            # SPY 12M 報酬
            spy_long_win = spy_price.iloc[i - lookback_long : i]
            if spy_long_win.isna().any():
                continue
            spy_long_ret = spy_long_win.iloc[-1] / spy_long_win.iloc[0] - 1.0

            # SPY 200MA
            spy_today = spy_price.iloc[i]
            spy_ma_today = spy_ma.iloc[i]

            # 如果 SPY 長期動能 <= 0 或 跌破 200MA → 全部持現金（權重=0）
            if spy_long_ret <= 0 or np.isnan(spy_ma_today) or spy_today < spy_ma_today:
                continue  # 這一天維持 0 權重

            # ===== 2. sector 動能（短 + 長） =====
            short_win = self.returns[assets].iloc[i - lookback_short : i]
            long_win  = self.returns[assets].iloc[i - lookback_long  : i]

            # 累積報酬
            short_mom = (1.0 + short_win).prod() - 1.0
            long_mom  = (1.0 + long_win).prod()  - 1.0

            # 只留 long_mom > 0 的資產（absolute momentum）
            eligible = long_mom[long_mom > 0].index
            if len(eligible) == 0:
                continue

            short_mom = short_mom[eligible]
            long_mom  = long_mom[eligible]

            # 結合短/長動能分數（簡單平均）
            score = 0.5 * short_mom + 0.5 * long_mom

            # 選前 top_k 檔
            top_assets = score.sort_values(ascending=False).index[:top_k]

            # ===== 3. 風險調整：用 1/vol 做 weighting =====
            vol_win = self.returns[top_assets].iloc[i - vol_lookback : i]
            sigma = vol_win.std()

            # 分數先平移成非負，再乘上 1/σ
            score_top = score[top_assets]
            score_shift = score_top - score_top.min() + 1e-6  # 確保 > 0
            inv_vol = 1.0 / sigma.replace(0, np.nan)
            combined = score_shift * inv_vol
            combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            if combined.sum() == 0:
                # fallback：等權
                w = np.ones(len(top_assets)) / len(top_assets)
            else:
                w = (combined / combined.sum()).values

            # ===== 4. 寫入這一天的權重 =====
            self.portfolio_weights.loc[date, top_assets] = w

        # 確保 SPY 權重一直是 0
        self.portfolio_weights[self.exclude] = 0.0
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
