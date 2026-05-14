from pathlib import Path
import yaml
import pandas as pd
import streamlit as st
from core.screener import run_screening, update_prime_tickers, load_tickers, OUTPUT, DATA

st.set_page_config(page_title='東証プライム Day0出来高スクリーナー', layout='wide')
st.title('東証プライム Day0出来高スクリーナー')
st.caption('yfinance版。投資判断は必ずご自身で確認してください。')

CFG_PATH = Path('settings.yaml')

def load_cfg():
    if CFG_PATH.exists():
        return yaml.safe_load(CFG_PATH.read_text(encoding='utf-8')) or {}
    return {}

cfg = load_cfg()
with st.sidebar:
    st.header('条件設定')
    cfg['lookback_days'] = st.number_input('取得日数', 60, 720, int(cfg.get('lookback_days',180)), 10)
    cfg['max_tickers'] = st.number_input('分析銘柄数（0=全件）', 0, 2000, int(cfg.get('max_tickers',200)), 50)
    cfg['volume_ratio_threshold'] = st.slider('出来高倍率しきい値', 0.5, 10.0, float(cfg.get('volume_ratio_threshold',1.5)), 0.1)
    cfg['use_market_cap_filter'] = st.checkbox('時価総額フィルタを使う', bool(cfg.get('use_market_cap_filter', True)))
    cfg['market_cap_min_oku'] = st.number_input('最低時価総額（億円）', 0, 10000, int(cfg.get('market_cap_min_oku',100)), 50)
    cfg['require_bullish'] = st.checkbox('陽線を必須にする', bool(cfg.get('require_bullish', True)))
    cfg['require_above_25ma'] = st.checkbox('終値>25MAを必須にする', bool(cfg.get('require_above_25ma', True)))
    cfg['top_n'] = st.number_input('表示上位数', 1, 100, int(cfg.get('top_n',10)), 1)
    if st.button('設定を保存'):
        CFG_PATH.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        st.success('保存しました')

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('東証プライム銘柄リストを更新'):
        with st.spinner('JPXリストを取得中...'):
            try:
                df_t = update_prime_tickers()
                st.success(f'{len(df_t)}銘柄を保存しました')
            except Exception as e:
                st.error(f'更新失敗: {e}')
with col2:
    if st.button('この条件でスクリーニング実行', type='primary'):
        prog = st.progress(0, text='開始します')
        def cb(v, text):
            prog.progress(min(float(v),1.0), text=text)
        with st.spinner('分析中...'):
            df, diag = run_screening(cfg, progress=cb)
        st.success(f'完了: {len(df)}件')
        st.session_state['last_df'] = df
        st.session_state['last_diag'] = diag
with col3:
    tickers = load_tickers()
    st.metric('登録銘柄数', len(tickers))

latest = OUTPUT / 'latest_results.csv'
diag_latest = OUTPUT / 'latest_diagnostics.csv'
if 'last_df' in st.session_state:
    df = st.session_state['last_df']
elif latest.exists():
    try: df = pd.read_csv(latest)
    except Exception: df = pd.DataFrame()
else:
    df = pd.DataFrame()

st.subheader('結果')
st.metric('抽出件数', len(df))
if not df.empty:
    st.dataframe(df, use_container_width=True)
    st.download_button('CSVをダウンロード', df.to_csv(index=False).encode('utf-8-sig'), 'screening_results.csv', 'text/csv')
else:
    st.info('条件に合う銘柄はありません。左側の条件を緩めて実行できます。')

st.subheader('診断')
if 'last_diag' in st.session_state:
    diag = st.session_state['last_diag']
elif diag_latest.exists():
    try: diag = pd.read_csv(diag_latest)
    except Exception: diag = pd.DataFrame()
else:
    diag = pd.DataFrame()
if not diag.empty:
    cols = ['取得','基礎計算','時価総額','出来高','陽線','25MA上','通過']
    st.json({c:int(diag[c].sum()) for c in cols if c in diag.columns})
    with st.expander('銘柄別診断を見る'):
        st.dataframe(diag, use_container_width=True)
