from __future__ import annotations
import io, os, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'output'
DATA.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

JPX_LIST_URL = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'

COLUMNS = [
    '推奨順位','抽出日','Day0日付','コード','銘柄名','ticker','終値','出来高',
    '20日平均出来高_急増除く','出来高倍率','RSI14','25MA','25日乖離率%',
    '時価総額_億円','パターン','スコア','エントリー目安','損切りライン',
    '利確+8%','利確+15%','利確+20%','株探','理由'
]

def load_tickers() -> pd.DataFrame:
    p = DATA / 'tickers.csv'
    if not p.exists():
        return pd.DataFrame(columns=['code','name','ticker'])
    df = pd.read_csv(p, dtype={'code': str})
    return df[['code','name','ticker']].dropna()

def update_prime_tickers() -> pd.DataFrame:
    # JPX公式「その他統計資料」の上場銘柄一覧Excelからプライム抽出
    r = requests.get(JPX_LIST_URL, timeout=30)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), dtype=str)
    cols = list(raw.columns)
    code_col = next((c for c in cols if 'コード' in str(c)), None)
    name_col = next((c for c in cols if '銘柄名' in str(c)), None)
    market_col = next((c for c in cols if '市場・商品区分' in str(c)), None)
    if not all([code_col, name_col, market_col]):
        raise RuntimeError('JPX銘柄リストの列名を認識できませんでした。')
    df = raw[raw[market_col].astype(str).str.contains('プライム', na=False)].copy()
    out = pd.DataFrame({
        'code': df[code_col].astype(str).str.extract(r'(\d{4})')[0],
        'name': df[name_col].astype(str),
    }).dropna()
    out['ticker'] = out['code'] + '.T'
    out = out.drop_duplicates('code').sort_values('code')
    out.to_csv(DATA / 'tickers.csv', index=False, encoding='utf-8-sig')
    return out

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yf.download(single ticker) sometimes returns MultiIndex with ticker level
        try:
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    return df

def volume_base_ex_spikes(vol: pd.Series, window: int = 20) -> pd.Series:
    # 当日を除いた過去20日平均。急増日は上位10%を除外して平均。
    def calc(x):
        s = pd.Series(x).dropna()
        if len(s) == 0:
            return np.nan
        q = s.quantile(0.9)
        s2 = s[s <= q]
        return s2.mean() if len(s2) else s.mean()
    return vol.shift(1).rolling(window).apply(calc, raw=False)

def pattern(row, hist: pd.DataFrame) -> str:
    close = float(row['Close'])
    recent_high = hist['Close'].shift(1).rolling(60).max().iloc[-1]
    ma75 = hist['Close'].rolling(75).mean().iloc[-1]
    if pd.notna(recent_high) and close >= recent_high:
        return 'A_レンジ上限ブレイクアウト'
    if pd.notna(ma75) and close < ma75 and row.get('RSI14', np.nan) < 45:
        return 'C_底値圏リバウンド'
    return 'Day0候補'

def score(row, pat: str, prev_close: float | None) -> int:
    s = 0
    vr = row.get('VolumeRatio', 0) or 0
    if vr >= 10: s += 30
    elif vr >= 5: s += 20
    elif vr >= 3: s += 10
    elif vr >= 1.5: s += 5
    if row['Close'] > row['Open']: s += 10
    if row['Close'] >= row['High'] * 0.98: s += 5
    if prev_close and row['Open'] > prev_close * 1.01: s += 5
    r = row.get('RSI14', np.nan)
    if pd.notna(r) and 50 <= r <= 65: s += 15
    elif pd.notna(r) and 45 <= r <= 70: s += 8
    if row.get('MA25_up', False): s += 10
    if row.get('Rebound5MA', False): s += 5
    if pat.startswith('A_'): s += 15
    elif pat.startswith('B_'): s += 20
    elif pat.startswith('C_'): s += 10
    return int(min(s, 100))

def analyze_one(code: str, name: str, ticker: str, cfg: dict) -> tuple[dict|None, dict]:
    diag = {'code': code, 'name': name, 'ticker': ticker, '取得': False, '基礎計算': False, '時価総額': False, '出来高': False, '陽線': False, '25MA上': False, '通過': False, '理由': ''}
    try:
        df = yf.download(ticker, period=f"{int(cfg.get('lookback_days',180))}d", progress=False, auto_adjust=False, threads=False)
        df = _normalize_yf(df, ticker)
        if df.empty or len(df) < 30:
            diag['理由'] = 'データ不足'
            return None, diag
        diag['取得'] = True
        df = df.dropna(subset=['Open','High','Low','Close','Volume']).copy()
        df['MA25'] = df['Close'].rolling(25).mean()
        df['MA25_prev'] = df['MA25'].shift(1)
        df['RSI14'] = rsi(df['Close'])
        df['VolBase20'] = volume_base_ex_spikes(df['Volume'])
        df['VolumeRatio'] = df['Volume'] / df['VolBase20']
        df['MA25_up'] = df['MA25'] > df['MA25_prev']
        df['MA5'] = df['Close'].rolling(5).mean()
        df['Rebound5MA'] = (df['Low'] <= df['MA5'] * 1.01) & (df['Close'] > df['MA5'])
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        if pd.isna(latest['MA25']) or pd.isna(latest['VolBase20']):
            diag['理由'] = '基礎指標不足'
            return None, diag
        diag['基礎計算'] = True
        try:
            mcap = yf.Ticker(ticker).fast_info.get('market_cap')
        except Exception:
            mcap = None
        mcap_oku = round(mcap / 100_000_000, 1) if mcap else None
        use_mcap = bool(cfg.get('use_market_cap_filter', True))
        min_oku = float(cfg.get('market_cap_min_oku', 100))
        diag['時価総額'] = (not use_mcap) or (mcap_oku is not None and mcap_oku >= min_oku)
        if not diag['時価総額']:
            diag['理由'] = '時価総額条件未満/取得不可'
            return None, diag
        vr_th = float(cfg.get('volume_ratio_threshold', 1.5))
        diag['出来高'] = float(latest['VolumeRatio']) >= vr_th
        if not diag['出来高']:
            diag['理由'] = '出来高倍率未満'
            return None, diag
        diag['陽線'] = bool(latest['Close'] > latest['Open'])
        if cfg.get('require_bullish', True) and not diag['陽線']:
            diag['理由'] = '陽線ではない'
            return None, diag
        diag['25MA上'] = bool(latest['Close'] > latest['MA25'])
        if cfg.get('require_above_25ma', True) and not diag['25MA上']:
            diag['理由'] = '25MA上ではない'
            return None, diag
        pat = pattern(latest, df)
        sc = score(latest, pat, float(prev['Close']))
        close = float(latest['Close'])
        result = {
            '抽出日': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'Day0日付': df.index[-1].strftime('%Y-%m-%d'),
            'コード': code,
            '銘柄名': name,
            'ticker': ticker,
            '終値': round(close, 2),
            '出来高': int(latest['Volume']),
            '20日平均出来高_急増除く': int(latest['VolBase20']),
            '出来高倍率': round(float(latest['VolumeRatio']), 2),
            'RSI14': round(float(latest['RSI14']), 1) if pd.notna(latest['RSI14']) else None,
            '25MA': round(float(latest['MA25']), 2),
            '25日乖離率%': round((close / float(latest['MA25']) - 1) * 100, 2),
            '時価総額_億円': mcap_oku,
            'パターン': pat,
            'スコア': sc,
            'エントリー目安': round(close, 2),
            '損切りライン': round(float(latest['Low']) * 0.99, 2),
            '利確+8%': round(close * 1.08, 2),
            '利確+15%': round(close * 1.15, 2),
            '利確+20%': round(close * 1.20, 2),
            '株探': f'https://kabutan.jp/stock/?code={code}',
            '理由': ''
        }
        diag['通過'] = True
        diag['理由'] = '通過'
        return result, diag
    except Exception as e:
        diag['理由'] = f'エラー: {type(e).__name__}: {e}'
        return None, diag

def run_screening(cfg: dict, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = load_tickers()
    max_t = int(cfg.get('max_tickers', 200) or 0)
    if max_t > 0:
        tickers = tickers.head(max_t)
    results, diags = [], []
    total = len(tickers)
    for i, row in tickers.iterrows():
        if progress:
            progress((len(diags)+1)/max(total,1), f"{len(diags)+1}/{total} {row['ticker']}")
        res, diag = analyze_one(str(row['code']).zfill(4), row['name'], row['ticker'], cfg)
        diags.append(diag)
        if res:
            results.append(res)
        time.sleep(0.02)
    df = pd.DataFrame(results, columns=[c for c in COLUMNS if c != '推奨順位'])
    if not df.empty:
        df = df.sort_values(['スコア','出来高倍率'], ascending=False).head(int(cfg.get('top_n',10))).reset_index(drop=True)
        df.insert(0, '推奨順位', range(1, len(df)+1))
    else:
        df = pd.DataFrame(columns=COLUMNS)
    diag_df = pd.DataFrame(diags)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(OUTPUT / 'latest_results.csv', index=False, encoding='utf-8-sig')
    diag_df.to_csv(OUTPUT / 'latest_diagnostics.csv', index=False, encoding='utf-8-sig')
    df.to_csv(OUTPUT / f'results_{ts}.csv', index=False, encoding='utf-8-sig')
    html = '<html><meta charset="utf-8"><body><h1>東証プライム Day0出来高スクリーナー</h1>' + df.to_html(index=False, escape=False) + '</body></html>'
    (OUTPUT / 'latest_report.html').write_text(html, encoding='utf-8')
    return df, diag_df
