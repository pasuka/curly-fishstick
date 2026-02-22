"""
网格交易回测工具 - Grid Trading Backtesting Tool
基于 Panel 框架，支持股票和ETF基金的网格交易策略回测
"""

import panel as pn
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Span, Label
from bokeh.models import NumeralTickFormatter, DatetimeTickFormatter
from bokeh.layouts import column as bk_column

pn.extension('tabulator', sizing_mode='stretch_width')

# ========================
# 数据获取模块
# ========================

def fetch_stock_data(symbol: str, start_date: str, end_date: str, interval: str = '5m') -> pd.DataFrame:
    """
    使用 akshare 获取A股/ETF的分钟级别数据。
    如果 akshare 不可用，则回退到 yfinance（适用于美股/港股等）。
    
    参数:
        symbol: 股票/ETF代码，如 '510300'（沪深ETF）或 'AAPL'（美股）
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        interval: 数据间隔，默认 '5m'（5分钟）
    返回:
        DataFrame: 包含 datetime, open, high, low, close, volume 列
    """
    df = pd.DataFrame()
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    requested_days = (end_dt - start_dt).days

    # 长区间分钟数据常受数据源窗口限制，自动降频以覆盖用户选择日期范围
    effective_interval = interval
    if requested_days > 60 and interval in {'1m', '5m', '15m', '30m'}:
        effective_interval = '60m'

    # ----- 尝试使用 akshare（A股/ETF） -----
    try:
        import akshare as ak

        # akshare 分钟数据接口：东方财富源
        # period 参数: "1", "5", "15", "30", "60"
        period_map = {'1m': '1', '5m': '5', '15m': '15', '30m': '30', '60m': '60'}
        ak_period = period_map.get(effective_interval, '5')

        # 尝试 A 股分钟数据
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=ak_period,
                start_date=start_date.replace('-', '') + " 09:30:00",
                end_date=end_date.replace('-', '') + " 15:00:00",
                adjust="qfq"  # 前复权
            )
        except Exception:
            # 尝试 ETF 分钟数据（基金接口）
            try:
                df = ak.fund_etf_hist_min_em(
                    symbol=symbol,
                    period=ak_period,
                    start_date=start_date.replace('-', '') + " 09:30:00",
                    end_date=end_date.replace('-', '') + " 15:00:00",
                    adjust="qfq"
                )
            except Exception:
                pass

        if not df.empty:
            # 统一列名
            col_map = {
                '时间': 'datetime', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
            }
            df = df.rename(columns=col_map)
            needed = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in needed if c in df.columns]]
            df['datetime'] = pd.to_datetime(df['datetime'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close']).reset_index(drop=True)
            return df

    except ImportError:
        pass

    # ----- 回退：使用 yfinance -----
    try:
        import yfinance as yf

        # yfinance 5m 数据最多保留 60 天
        end_dt_with_bound = end_dt + timedelta(days=1)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_dt, end=end_dt_with_bound, interval=effective_interval)

        if not df.empty:
            df = df.reset_index()
            rename_map = {
                'Datetime': 'datetime', 'Date': 'datetime',
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume',
            }
            df = df.rename(columns=rename_map)
            needed = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in needed if c in df.columns]]
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_localize(None)
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close']).reset_index(drop=True)
            return df

    except ImportError:
        pass

    # ----- 全部失败：生成模拟数据 -----
    if df.empty:
        df = _generate_simulated_data(symbol, start_date, end_date)

    return df


def _generate_simulated_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    当无法获取真实数据时，生成模拟的 5 分钟 K 线数据用于演示。
    """
    np.random.seed(hash(symbol) % 2**31)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    trading_days = pd.bdate_range(start=start_dt, end=end_dt)

    records = []
    # 模拟初始价格
    price = np.random.uniform(3, 50)

    for day in trading_days:
        # A股交易时段：9:30-11:30, 13:00-15:00
        am_times = pd.date_range(
            start=day + pd.Timedelta(hours=9, minutes=30),
            end=day + pd.Timedelta(hours=11, minutes=25),
            freq='5min'
        )
        pm_times = pd.date_range(
            start=day + pd.Timedelta(hours=13, minutes=0),
            end=day + pd.Timedelta(hours=14, minutes=55),
            freq='5min'
        )
        bar_times = am_times.append(pm_times)

        for t in bar_times:
            change = price * np.random.normal(0, 0.003)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, price * 0.001))
            l = min(o, c) - abs(np.random.normal(0, price * 0.001))
            v = int(np.random.uniform(1000, 50000))
            records.append({
                'datetime': t,
                'open': round(o, 3),
                'high': round(h, 3),
                'low': round(l, 3),
                'close': round(c, 3),
                'volume': v,
            })
            price = c

    df = pd.DataFrame(records)
    return df


# ========================
# 网格交易回测引擎
# ========================

class GridTradingEngine:
    """网格交易回测引擎"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        sell_pct: float = 3.0,        # 上涨卖出百分比
        buy_pct: float = 3.0,         # 下跌买入百分比
        commission: float = 0.03,     # 手续费百分比
        min_shares: int | None = 0,   # 最少持股数（None=不限制）
        max_shares: int | None = 10000,  # 最大持股数（None=不限制）
        grid_lot: int = 100,          # 每次交易手数（股）
        price_upper: float = None,    # 网格价格上限
        price_lower: float = None,    # 网格价格下限
    ):
        self.data = data.copy()
        if 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data = self.data.sort_values('datetime')
        self.data = self.data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.sell_pct = sell_pct / 100.0
        self.buy_pct = buy_pct / 100.0
        self.commission = commission / 100.0
        self.min_shares = min_shares
        self.max_shares = max_shares
        self.grid_lot = grid_lot
        self.price_upper = price_upper
        self.price_lower = price_lower

        # 状态
        self.cash = initial_capital
        self.shares = 0
        self.last_trade_price = None
        self.trades = []          # 交易记录
        self.portfolio_values = []  # 每个时间点的组合价值

    def run(self):
        """执行回测"""
        for i, row in self.data.iterrows():
            price = row['close']
            dt = row['datetime']

            # 判断价格是否在网格区间内
            if self.price_upper and price > self.price_upper:
                self._record_portfolio(dt, price)
                continue
            if self.price_lower and price < self.price_lower:
                self._record_portfolio(dt, price)
                continue

            # 首笔建仓
            if self.last_trade_price is None:
                qty = self._calc_initial_buy_qty(price)
                if qty > 0:
                    self._execute_buy(dt, price, qty)
                self._record_portfolio(dt, price)
                continue

            # 网格触发判断
            price_change_pct = (price - self.last_trade_price) / self.last_trade_price

            if price_change_pct >= self.sell_pct:
                # 触发卖出
                min_shares = 0 if self.min_shares is None else self.min_shares
                qty = min(self.grid_lot, self.shares - min_shares)
                if qty > 0 and self.shares - qty >= min_shares:
                    self._execute_sell(dt, price, qty)
            elif price_change_pct <= -self.buy_pct:
                # 触发买入
                qty = self._calc_buy_qty(price)
                if qty > 0:
                    self._execute_buy(dt, price, qty)

            self._record_portfolio(dt, price)

    def _calc_buy_qty(self, price: float) -> int:
        """计算可买入数量"""
        affordable = int(self.cash / (price * (1 + self.commission)))
        # 取 grid_lot 的整数倍
        qty = (affordable // self.grid_lot) * self.grid_lot
        qty = min(qty, self.grid_lot)
        # 检查最大持股限制
        if self.max_shares is not None and self.shares + qty > self.max_shares:
            qty = self.max_shares - self.shares
            qty = (qty // self.grid_lot) * self.grid_lot
        return max(0, qty)

    def _calc_initial_buy_qty(self, price: float) -> int:
        """计算首笔建仓数量（使用现金的 50%）"""
        budget = self.cash * 0.5
        affordable = int(budget / (price * (1 + self.commission)))
        qty = (affordable // self.grid_lot) * self.grid_lot
        if self.max_shares is not None and self.shares + qty > self.max_shares:
            qty = self.max_shares - self.shares
            qty = (qty // self.grid_lot) * self.grid_lot
        return max(0, qty)

    def _execute_buy(self, dt, price, qty):
        cost = price * qty * (1 + self.commission)
        self.cash -= cost
        self.shares += qty
        self.last_trade_price = price
        self.trades.append({
            '时间': dt,
            '类型': '买入',
            '价格': round(price, 4),
            '数量': qty,
            '金额': round(cost, 2),
            '手续费': round(price * qty * self.commission, 2),
            '持仓': self.shares,
            '现金余额': round(self.cash, 2),
        })

    def _execute_sell(self, dt, price, qty):
        revenue = price * qty * (1 - self.commission)
        self.cash += revenue
        self.shares -= qty
        self.last_trade_price = price
        self.trades.append({
            '时间': dt,
            '类型': '卖出',
            '价格': round(price, 4),
            '数量': qty,
            '金额': round(revenue, 2),
            '手续费': round(price * qty * self.commission, 2),
            '持仓': self.shares,
            '现金余额': round(self.cash, 2),
        })

    def _record_portfolio(self, dt, price):
        total = self.cash + self.shares * price
        self.portfolio_values.append({
            'datetime': dt,
            'portfolio_value': total,
            'cash': self.cash,
            'stock_value': self.shares * price,
            'shares': self.shares,
            'price': price,
        })

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                '时间', '类型', '价格', '数量', '金额', '手续费', '持仓', '现金余额'
            ])
        return pd.DataFrame(self.trades)

    def get_portfolio_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_values)

    def get_summary(self) -> dict:
        """获取回测摘要"""
        if not self.portfolio_values:
            return {}
        final_value = self.portfolio_values[-1]['portfolio_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        total_trades = len(self.trades)
        buy_trades = sum(1 for t in self.trades if t['类型'] == '买入')
        sell_trades = sum(1 for t in self.trades if t['类型'] == '卖出')
        total_commission = sum(t['手续费'] for t in self.trades)

        # 最大回撤
        pv = [p['portfolio_value'] for p in self.portfolio_values]
        peak = pv[0]
        max_dd = 0
        for v in pv:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            '初始资金': f"¥{self.initial_capital:,.2f}",
            '最终资金': f"¥{final_value:,.2f}",
            '总收益率': f"{total_return:.2f}%",
            '最大回撤': f"{max_dd * 100:.2f}%",
            '总交易次数': total_trades,
            '买入次数': buy_trades,
            '卖出次数': sell_trades,
            '总手续费': f"¥{total_commission:,.2f}",
            '最终持仓': f"{self.shares} 股",
            '现金余额': f"¥{self.cash:,.2f}",
        }


def calculate_backtest_metrics(engine: GridTradingEngine) -> dict:
    """计算用于参数优化的数值指标。"""
    if not engine.portfolio_values:
        return {
            '总收益率(%)': -999.0,
            '最大回撤(%)': 999.0,
            '交易次数': 0,
            '手续费(¥)': 0.0,
            '最终资金(¥)': engine.initial_capital,
        }

    final_value = engine.portfolio_values[-1]['portfolio_value']
    total_return = (final_value - engine.initial_capital) / engine.initial_capital * 100

    pv = [p['portfolio_value'] for p in engine.portfolio_values]
    peak = pv[0]
    max_dd = 0
    for v in pv:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    total_commission = sum(t['手续费'] for t in engine.trades)
    return {
        '总收益率(%)': round(total_return, 4),
        '最大回撤(%)': round(max_dd * 100, 4),
        '交易次数': len(engine.trades),
        '手续费(¥)': round(total_commission, 2),
        '最终资金(¥)': round(final_value, 2),
    }


# ========================
# 图表绘制模块
# ========================

def create_kline_chart(data: pd.DataFrame, trades_df: pd.DataFrame, symbol: str):
    """创建K线图 + 交易标记（使用日线聚合展示）"""

    # 将5分钟数据聚合为日线，方便K线展示
    df = data.copy()
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'datetime': 'first',
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])

    inc = daily[daily.close >= daily.open]
    dec = daily[daily.close < daily.open]

    w = 12 * 60 * 60 * 1000  # 半日宽度（毫秒）

    p = figure(
        title=f"{symbol} K线图 & 网格交易信号",
        x_axis_type='datetime',
        width=960, height=420,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_drag="pan",
    )
    p.xaxis.formatter = DatetimeTickFormatter(days='%Y-%m-%d', months='%Y-%m')

    # 涨（红）
    p.segment(inc.date, inc.high, inc.date, inc.low, color="#e74c3c")
    p.vbar(inc.date, w, inc.open, inc.close, fill_color="#e74c3c",
           line_color="#e74c3c", alpha=0.8)
    # 跌（绿）
    p.segment(dec.date, dec.high, dec.date, dec.low, color="#2ecc71")
    p.vbar(dec.date, w, dec.open, dec.close, fill_color="#2ecc71",
           line_color="#2ecc71", alpha=0.8)

    # 交易标记
    if not trades_df.empty:
        buys = trades_df[trades_df['类型'] == '买入'].copy()
        sells = trades_df[trades_df['类型'] == '卖出'].copy()

        if not buys.empty:
            buys['dt'] = pd.to_datetime(buys['时间'])
            buy_src = ColumnDataSource(data=dict(
                x=buys['dt'], y=buys['价格'],
                qty=buys['数量'], amt=buys['金额'],
            ))
            p.scatter(
                'x', 'y', source=buy_src,
                marker='triangle',
                size=12, color='#3498db', alpha=0.9,
                legend_label='买入'
            )

        if not sells.empty:
            sells['dt'] = pd.to_datetime(sells['时间'])
            sell_src = ColumnDataSource(data=dict(
                x=sells['dt'], y=sells['价格'],
                qty=sells['数量'], amt=sells['金额'],
            ))
            p.scatter(
                'x', 'y', source=sell_src,
                marker='inverted_triangle',
                size=12, color='#e67e22', alpha=0.9,
                legend_label='卖出'
            )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.yaxis.axis_label = "价格"
    p.xaxis.axis_label = "日期"
    return p


def create_portfolio_chart(portfolio_df: pd.DataFrame, initial_capital: float):
    """创建资产价值曲线图"""
    if portfolio_df.empty:
        return figure(title="资产曲线（无数据）", width=960, height=260)

    # 聚合到日线
    pdf = portfolio_df.copy()
    pdf['date'] = pdf['datetime'].dt.date
    daily_pv = pdf.groupby('date').agg({
        'portfolio_value': 'last',
        'cash': 'last',
        'stock_value': 'last',
    }).reset_index()
    daily_pv['date'] = pd.to_datetime(daily_pv['date'])

    p = figure(
        title="账户总资产变化曲线",
        x_axis_type='datetime',
        width=960, height=260,
        tools="pan,wheel_zoom,box_zoom,reset",
    )

    p.line(daily_pv['date'], daily_pv['portfolio_value'],
           line_width=2, color='#2980b9', legend_label='总资产')
    p.line(daily_pv['date'], daily_pv['cash'],
           line_width=1.5, color='#27ae60', alpha=0.7, legend_label='现金')
    p.line(daily_pv['date'], daily_pv['stock_value'],
           line_width=1.5, color='#e74c3c', alpha=0.7, legend_label='持仓市值')

    # 初始资金参考线
    init_line = Span(
        location=initial_capital, dimension='width',
        line_color='gray', line_dash='dashed', line_width=1,
    )
    p.add_layout(init_line)
    p.add_layout(Label(
        x=daily_pv['date'].iloc[0], y=initial_capital,
        text=f'初始资金: ¥{initial_capital:,.0f}',
        text_font_size='9pt', text_color='gray',
        x_offset=5, y_offset=5,
    ))

    p.yaxis.formatter = NumeralTickFormatter(format='¥0,0')
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.yaxis.axis_label = "金额"
    return p


# ========================
# Panel UI 构建
# ========================

def build_app():
    """构建 Panel 应用"""

    # ---------- 左侧输入面板 ----------
    symbol_input = pn.widgets.TextInput(
        name='📈 股票/ETF代码',
        value='510300',
        placeholder='例: 510300, 159915, AAPL',
        width=260,
    )
    start_date_input = pn.widgets.DatePicker(
        name='📅 开始日期',
        value=(datetime.now() - timedelta(days=180)).date(),
        width=260,
    )
    end_date_input = pn.widgets.DatePicker(
        name='📅 结束日期',
        value=datetime.now().date(),
        width=260,
    )

    pn.widgets.StaticText(name='', value='── 网格参数 ──')

    sell_pct_input = pn.widgets.FloatInput(
        name='📊 上涨卖出百分比 (%)',
        value=3.0, step=0.1, start=0.1, end=50.0,
        width=260,
    )
    buy_pct_input = pn.widgets.FloatInput(
        name='📊 下跌买入百分比 (%)',
        value=3.0, step=0.1, start=0.1, end=50.0,
        width=260,
    )
    commission_input = pn.widgets.FloatInput(
        name='💰 手续费 (%)',
        value=0.03, step=0.01, start=0.0, end=5.0,
        width=260,
    )
    grid_lot_input = pn.widgets.IntInput(
        name='📦 每格交易数量（股）',
        value=100, step=100, start=1,
        width=260,
    )
    min_shares_input = pn.widgets.IntInput(
        name='📉 最少持股数',
        value=0, step=100, start=0,
        width=260,
    )
    min_shares_enable = pn.widgets.Checkbox(
        name='启用最少持股限制',
        value=False,
        width=260,
    )
    max_shares_input = pn.widgets.IntInput(
        name='📈 最大持股数',
        value=10000, step=100, start=100,
        width=260,
    )
    max_shares_enable = pn.widgets.Checkbox(
        name='启用最大持股限制',
        value=False,
        width=260,
    )
    price_lower_input = pn.widgets.FloatInput(
        name='🔻 网格价格下限（0=不限）',
        value=0, step=0.1, start=0,
        width=260,
    )
    price_upper_input = pn.widgets.FloatInput(
        name='🔺 网格价格上限（0=不限）',
        value=0, step=0.1, start=0,
        width=260,
    )
    initial_capital_input = pn.widgets.FloatInput(
        name='🏦 初始资金 (¥)',
        value=100000.0, step=10000, start=1000,
        width=260,
    )
    run_button = pn.widgets.Button(
        name='🚀 开始回测',
        button_type='primary',
        width=260,
        height=45,
    )
    optimize_button = pn.widgets.Button(
        name='⚡ 自动优化参数（MVP）',
        button_type='warning',
        width=260,
        height=40,
    )
    optimize_topn_input = pn.widgets.IntInput(
        name='📌 优化结果 Top-N',
        value=5, step=1, start=1, end=20,
        width=260,
    )
    apply_best_button = pn.widgets.Button(
        name='✅ 应用最佳参数并回测',
        button_type='success',
        width=260,
        height=40,
        disabled=True,
    )

    status_text = pn.pane.Alert(
        '请设置参数后点击「开始回测」', alert_type='info', width=260,
    )
    optimize_status_text = pn.pane.Alert(
        '可点击「自动优化参数」搜索更优组合', alert_type='info', width=260,
    )

    sidebar = pn.Column(
        pn.pane.Markdown("# 🔧 参数设置", styles={'color': '#2c3e50'}),
        pn.layout.Divider(),
        symbol_input,
        start_date_input,
        end_date_input,
        pn.layout.Divider(),
        pn.pane.Markdown("### ⚙️ 网格交易参数"),
        sell_pct_input,
        buy_pct_input,
        commission_input,
        grid_lot_input,
        min_shares_enable,
        min_shares_input,
        max_shares_enable,
        max_shares_input,
        price_lower_input,
        price_upper_input,
        initial_capital_input,
        pn.layout.Divider(),
        run_button,
        pn.layout.Divider(),
        pn.pane.Markdown("### 🧠 参数优化（MVP）"),
        optimize_topn_input,
        optimize_button,
        apply_best_button,
        optimize_status_text,
        status_text,
        width=300,
        styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px'},
    )

    # ---------- 右侧结果面板 ----------
    summary_pane = pn.pane.Markdown("", styles={'font-size': '14px'})
    chart_pane = pn.Column(
        pn.pane.Markdown("### 📊 点击「开始回测」查看结果", align='center'),
    )
    optimize_result_pane = pn.Column(
        pn.pane.Markdown("### 🧪 参数优化结果（尚未执行）")
    )
    trades_table_pane = pn.Column()
    optimization_state = {'best': None, 'results': None}

    def _build_engine_with_params(data, sell_pct, buy_pct, grid_lot):
        p_upper = price_upper_input.value if price_upper_input.value > 0 else None
        p_lower = price_lower_input.value if price_lower_input.value > 0 else None
        min_shares = min_shares_input.value if min_shares_enable.value else None
        max_shares = max_shares_input.value if max_shares_enable.value else None
        return GridTradingEngine(
            data=data,
            initial_capital=initial_capital_input.value,
            sell_pct=sell_pct,
            buy_pct=buy_pct,
            commission=commission_input.value,
            min_shares=min_shares,
            max_shares=max_shares,
            grid_lot=grid_lot,
            price_upper=p_upper,
            price_lower=p_lower,
        )

    def optimize_params(event):
        optimize_status_text.object = '⏳ 正在执行参数优化，请稍候...'
        optimize_status_text.alert_type = 'warning'
        try:
            symbol = symbol_input.value.strip()
            start_str = start_date_input.value.strftime('%Y-%m-%d')
            end_str = end_date_input.value.strftime('%Y-%m-%d')
            data = fetch_stock_data(symbol, start_str, end_str, interval='5m')
            if data.empty:
                optimize_status_text.object = '❌ 优化失败：未获取到数据'
                optimize_status_text.alert_type = 'danger'
                return

            sell_base = float(sell_pct_input.value)
            buy_base = float(buy_pct_input.value)
            grid_base = int(grid_lot_input.value)
            grid_step = max(1, int(getattr(grid_lot_input, 'step', 1) or 1))

            sell_candidates = sorted({
                max(0.1, round(sell_base - 1.0, 2)),
                max(0.1, round(sell_base - 0.5, 2)),
                round(sell_base, 2),
                round(sell_base + 0.5, 2),
                round(sell_base + 1.0, 2),
            })
            buy_candidates = sorted({
                max(0.1, round(buy_base - 1.0, 2)),
                max(0.1, round(buy_base - 0.5, 2)),
                round(buy_base, 2),
                round(buy_base + 0.5, 2),
                round(buy_base + 1.0, 2),
            })
            grid_candidates = sorted({
                max(grid_step, grid_base - grid_step),
                grid_base,
                grid_base + grid_step,
            })

            rows = []
            for sell_pct, buy_pct, grid_lot in product(
                sell_candidates, buy_candidates, grid_candidates
            ):
                engine = _build_engine_with_params(data, sell_pct, buy_pct, int(grid_lot))
                engine.run()
                metrics = calculate_backtest_metrics(engine)
                rows.append({
                    '卖出阈值(%)': sell_pct,
                    '买入阈值(%)': buy_pct,
                    '每格交易股数': int(grid_lot),
                    **metrics,
                })

            if not rows:
                optimize_status_text.object = '❌ 未生成优化结果'
                optimize_status_text.alert_type = 'danger'
                return

            results_df = pd.DataFrame(rows).sort_values(
                by=['总收益率(%)', '最大回撤(%)', '交易次数'],
                ascending=[False, True, True],
            ).reset_index(drop=True)

            optimization_state['results'] = results_df
            optimization_state['best'] = results_df.iloc[0].to_dict()
            apply_best_button.disabled = False

            top_n = max(1, int(optimize_topn_input.value))
            top_df = results_df.head(top_n).copy()
            optimize_result_pane.clear()
            optimize_result_pane.append(
                pn.pane.Markdown(
                    f"### 🧪 参数优化结果（Top {min(top_n, len(results_df))} / {len(results_df)} 组合）"
                )
            )
            optimize_result_pane.append(
                pn.widgets.Tabulator(
                    top_df,
                    name='优化结果',
                    pagination='remote',
                    page_size=min(20, max(5, top_n)),
                    layout='fit_columns',
                    theme='bootstrap',
                    height=300,
                )
            )

            best = optimization_state['best']
            optimize_status_text.object = (
                f"✅ 优化完成：最佳参数 卖出{best['卖出阈值(%)']}% / "
                f"买入{best['买入阈值(%)']}% / 每格{int(best['每格交易股数'])}股"
            )
            optimize_status_text.alert_type = 'success'

        except Exception as e:
            optimize_status_text.object = f'❌ 优化错误: {str(e)}'
            optimize_status_text.alert_type = 'danger'
            import traceback
            traceback.print_exc()

    def apply_best_params(event):
        best = optimization_state.get('best')
        if not best:
            optimize_status_text.object = '⚠️ 请先执行一次参数优化'
            optimize_status_text.alert_type = 'warning'
            return

        sell_pct_input.value = float(best['卖出阈值(%)'])
        buy_pct_input.value = float(best['买入阈值(%)'])
        grid_lot_input.value = int(best['每格交易股数'])
        optimize_status_text.object = '✅ 已回填最佳参数，正在自动回测...'
        optimize_status_text.alert_type = 'success'
        run_backtest(None)

    # ---------- 回测逻辑 ----------
    def run_backtest(event):
        status_text.object = '⏳ 正在获取数据...'
        status_text.alert_type = 'warning'

        try:
            symbol = symbol_input.value.strip()
            start_str = start_date_input.value.strftime('%Y-%m-%d')
            end_str = end_date_input.value.strftime('%Y-%m-%d')

            # 获取数据
            data = fetch_stock_data(symbol, start_str, end_str, interval='5m')
            if data.empty:
                status_text.object = '❌ 未获取到数据，请检查代码和日期'
                status_text.alert_type = 'danger'
                return

            status_text.object = f'⏳ 获取到 {len(data)} 条数据，正在回测...'

            # 价格区间
            p_upper = price_upper_input.value if price_upper_input.value > 0 else None
            p_lower = price_lower_input.value if price_lower_input.value > 0 else None
            min_shares = min_shares_input.value if min_shares_enable.value else None
            max_shares = max_shares_input.value if max_shares_enable.value else None

            # 初始化引擎
            engine = GridTradingEngine(
                data=data,
                initial_capital=initial_capital_input.value,
                sell_pct=sell_pct_input.value,
                buy_pct=buy_pct_input.value,
                commission=commission_input.value,
                min_shares=min_shares,
                max_shares=max_shares,
                grid_lot=grid_lot_input.value,
                price_upper=p_upper,
                price_lower=p_lower,
            )

            # 执行回测
            engine.run()

            # 获取结果
            trades_df = engine.get_trades_df()
            portfolio_df = engine.get_portfolio_df()
            summary = engine.get_summary()

            # 展示层按右侧开始/结束日期过滤
            display_start_dt = pd.to_datetime(start_date_input.value)
            display_end_dt = pd.to_datetime(end_date_input.value) + timedelta(days=1)
            data_display = data[
                (data['datetime'] >= display_start_dt) & (data['datetime'] < display_end_dt)
            ].copy()
            if trades_df.empty:
                trades_df_chart = trades_df.copy()
                trades_df_display = trades_df.copy()
            else:
                trades_time = pd.to_datetime(trades_df['时间'])
                mask = (trades_time >= display_start_dt) & (trades_time < display_end_dt)
                trades_df_chart = trades_df[mask].copy()
                trades_df_display = trades_df_chart.copy()

            # ---- 更新摘要 ----
            summary_md = "## 📋 回测摘要\n\n"
            summary_md += "| 指标 | 数值 |\n|------|------|\n"
            for k, v in summary.items():
                summary_md += f"| **{k}** | {v} |\n"
            summary_md += f"\n*数据区间: {start_str} ~ {end_str}，共 {len(data)} 条5分钟K线*"
            if not data.empty:
                actual_start = pd.to_datetime(data['datetime'].min()).strftime('%Y-%m-%d %H:%M')
                actual_end = pd.to_datetime(data['datetime'].max()).strftime('%Y-%m-%d %H:%M')
                summary_md += f"\n*实际返回数据范围: {actual_start} ~ {actual_end}*"
            summary_pane.object = summary_md

            # ---- 更新图表 ----
            kline = create_kline_chart(data_display, trades_df_chart, symbol)
            portfolio_chart = create_portfolio_chart(portfolio_df, initial_capital_input.value)

            chart_pane.clear()
            chart_pane.append(pn.pane.Bokeh(kline, sizing_mode='stretch_width'))
            chart_pane.append(pn.pane.Bokeh(portfolio_chart, sizing_mode='stretch_width'))

            # ---- 更新交易记录表 ----
            trades_table_pane.clear()
            if not trades_df_display.empty:
                trades_df_display['时间'] = trades_df_display['时间'].astype(str)

                tabulator = pn.widgets.Tabulator(
                    trades_df_display,
                    name='交易记录',
                    page_size=20,
                    pagination='remote',
                    layout='fit_columns',
                    theme='bootstrap',
                    height=400,
                    frozen_columns=['时间'],
                    formatters={
                        '价格': {'type': 'money', 'precision': 4},
                        '金额': {'type': 'money', 'precision': 2},
                        '手续费': {'type': 'money', 'precision': 2},
                        '现金余额': {'type': 'money', 'precision': 2},
                    },
                    text_align={
                        '类型': 'center',
                        '数量': 'right',
                        '价格': 'right',
                        '金额': 'right',
                    },
                )
                trades_table_pane.append(
                    pn.pane.Markdown(
                        f"## 📝 交易记录（{start_str} ~ {end_str}，共 {len(trades_df_display)} 笔 / 总计 {len(trades_df)} 笔）"
                    )
                )
                trades_table_pane.append(tabulator)

                # CSV 下载
                csv_buf = trades_df_display.to_csv(index=False).encode('utf-8-sig')
                file_download = pn.widgets.FileDownload(
                    csv_buf,
                    filename=f'trades_{symbol}_{start_str}_{end_str}.csv',
                    button_type='success',
                    label='📥 下载交易记录 CSV',
                )
                trades_table_pane.append(file_download)
            else:
                trades_table_pane.append(
                    pn.pane.Alert("所选时间范围内未产生交易记录，请调整参数或日期区间", alert_type='warning')
                )

            status_text.object = f'✅ 回测完成！共 {len(trades_df)} 笔交易'
            status_text.alert_type = 'success'

        except Exception as e:
            status_text.object = f'❌ 错误: {str(e)}'
            status_text.alert_type = 'danger'
            import traceback
            traceback.print_exc()

    run_button.on_click(run_backtest)
    optimize_button.on_click(optimize_params)
    apply_best_button.on_click(apply_best_params)

    # ---------- 主布局 ----------
    main_content = pn.Column(
        summary_pane,
        optimize_result_pane,
        chart_pane,
        trades_table_pane,
        sizing_mode='stretch_width',
        styles={'padding': '10px'},
    )

    template_kwargs = {
        'title': '📈 网格交易回测工具 - Grid Trading Backtester',
        'sidebar': [sidebar],
        'main': [main_content],
        'sidebar_width': 320,
        'header_background': '#2c3e50',
        'accent_base_color': '#3498db',
    }
    while True:
        try:
            template = pn.template.MaterialTemplate(**template_kwargs)
            break
        except TypeError as e:
            err = str(e)
            if "unexpected keyword argument" not in err:
                raise
            bad_arg = err.split("'")[1]
            if bad_arg not in template_kwargs:
                raise
            template_kwargs.pop(bad_arg)

    return template


# ========================
# 启动应用
# ========================
app = build_app()
app.servable()

if __name__ == '__main__':
    # 运行: python grid_trading_backtest.py
    # 或:   panel serve grid_trading_backtest.py --show --autoreload
    import argparse

    parser = argparse.ArgumentParser(description='网格交易回测工具')
    parser.add_argument('--port', type=int, default=5006, help='服务端口，默认 5006')
    args = parser.parse_args()

    try:
        pn.serve(app, port=args.port, show=True, title='网格交易回测')
    except OSError as e:
        err = str(e)
        if '10048' in err:
            print(f'端口 {args.port} 被占用，自动切换到随机可用端口...')
            pn.serve(app, port=0, show=True, title='网格交易回测')
        else:
            raise