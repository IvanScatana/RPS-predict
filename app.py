import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# ========================
# Конфигурация
# ========================
DATA_PATH = 'rps_data.csv'

MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}
MOVE_EMOJI = {"Камень": "✊", "Ножницы": "✌️", "Бумага": "✋"}

# Базовые колонки + динамические prev3..prev6
BASE_COLS = [
    'match_id', 'round', 'player_name', 'win_category',
    'opp_match_wins', 'opp_match_winrate', 'stake',
    'opp_move', 'my_move', 'outcome',
    'score_me_before', 'score_opp_before', 'score_diff', 'streak_draws',
    'prev_opp_move', 'prev_my_move', 'prev_outcome',
    'prev2_opp_move', 'prev2_my_move', 'prev2_outcome'
]
EXTRA_COLS = []
for shift in [3,4,5,6]:
    EXTRA_COLS.extend([f'prev{shift}_opp_move', f'prev{shift}_my_move', f'prev{shift}_outcome'])
EXPECTED_COLS = BASE_COLS + EXTRA_COLS + ['is_last_round']

# ========================
# Вспомогательные функции
# ========================
def compute_win_category(wins):
    if wins == -1:
        return 'unknown'
    elif wins <= 5:
        return '<=5'
    elif wins <= 20:
        return '6-20'
    elif wins <= 100:
        return '21-100'
    else:
        return '>100'

def get_outcome(my_move, opp_move):
    if my_move == opp_move:
        return 'draw'
    if (my_move == 'К' and opp_move == 'Н') or (my_move == 'Н' and opp_move == 'Б') or (my_move == 'Б' and opp_move == 'К'):
        return 'win'
    return 'loss'

# -------------------- Вероятностные таблицы с подсчётом --------------------
def prepare_prob_r(df, round_num):
    df_r = df[df['round'] == round_num].copy()
    if df_r.empty:
        return pd.DataFrame()
    group_cols = ['stake', 'win_category']
    for i in range(1, round_num):
        for suffix in ['_outcome', '_my_move', '_opp_move']:
            col = f'prev{i}{suffix}'
            if col in df_r.columns:
                group_cols.append(col)
    counts = df_r.groupby(group_cols)['opp_move'].value_counts().reset_index(name='count')
    counts['prob'] = counts.groupby(group_cols)['count'].transform(lambda x: x / x.sum())
    return counts

def get_optimal_move_r(stake, win_category, outcomes, my_moves, opp_moves, prob_r):
    if prob_r.empty:
        return None, 0, 0
    mask = (prob_r['stake'] == stake) & (prob_r['win_category'] == win_category)
    for i, (outc, my_m, opp_m) in enumerate(zip(outcomes, my_moves, opp_moves), start=1):
        col_outc = f'prev{i}_outcome'
        col_my = f'prev{i}_my_move'
        col_opp = f'prev{i}_opp_move'
        if col_outc in prob_r.columns:
            mask &= (prob_r[col_outc] == outc)
        if col_my in prob_r.columns:
            mask &= (prob_r[col_my] == my_m)
        if col_opp in prob_r.columns:
            mask &= (prob_r[col_opp] == opp_m)
    subset = prob_r[mask]
    if len(subset) == 0:
        return None, 0, 0
    p_k = subset[subset['opp_move'] == 'К']['prob'].sum()
    p_n = subset[subset['opp_move'] == 'Н']['prob'].sum()
    p_b = subset[subset['opp_move'] == 'Б']['prob'].sum()
    exp_k = p_n - p_b
    exp_n = p_b - p_k
    exp_b = p_k - p_n
    if exp_k >= exp_n and exp_k >= exp_b:
        best_move = 'К'
    elif exp_n >= exp_b:
        best_move = 'Н'
    else:
        best_move = 'Б'
    best_row = subset.loc[subset['prob'].idxmax()]
    confidence = best_row['prob']
    support = subset['count'].sum()
    return best_move, confidence, support

# -------------------- Обёртки для раундов 1-7 --------------------
def get_optimal_move_r1(stake, win_category, df):
    sub = df[(df["round"] == 1) & (df["stake"] == stake) & (df["win_category"] == win_category)]
    if len(sub) == 0:
        sub = df[(df["round"] == 1) & (df["win_category"] == win_category)]
        if len(sub) == 0:
            sub = df[(df["round"] == 1) & (df["stake"] == stake)]
            if len(sub) == 0:
                return 'К', 0, 0
    counts = sub["opp_move"].value_counts()
    total = len(sub)
    p_k = counts.get('К', 0) / total
    p_n = counts.get('Н', 0) / total
    p_b = counts.get('Б', 0) / total
    exp_k = p_n - p_b
    exp_n = p_b - p_k
    exp_b = p_k - p_n
    if exp_k >= exp_n and exp_k >= exp_b:
        best = 'К'
    elif exp_n >= exp_b:
        best = 'Н'
    else:
        best = 'Б'
    best_count = counts.get(best, 0)
    confidence = best_count / total if total > 0 else 0
    return best, confidence, total

def get_optimal_move_r2(stake, win_category, outc1, my1, opp1, prob_r2, _):
    return get_optimal_move_r(stake, win_category, [outc1], [my1], [opp1], prob_r2)

def get_optimal_move_r3(stake, win_category, outc2, my2, opp2, outc1, my1, opp1, prob_r3, _):
    return get_optimal_move_r(stake, win_category, [outc2, outc1], [my2, my1], [opp2, opp1], prob_r3)

def get_optimal_move_r4(stake, win_category, outc3, my3, opp3, outc2, my2, opp2, outc1, my1, opp1, prob_r4):
    return get_optimal_move_r(stake, win_category, [outc3, outc2, outc1], [my3, my2, my1], [opp3, opp2, opp1], prob_r4)

def get_optimal_move_r5(stake, win_category, outc4, my4, opp4, outc3, my3, opp3, outc2, my2, opp2, outc1, my1, opp1, prob_r5):
    return get_optimal_move_r(stake, win_category, [outc4, outc3, outc2, outc1], [my4, my3, my2, my1], [opp4, opp3, opp2, opp1], prob_r5)

def get_optimal_move_r6(stake, win_category, outc5, my5, opp5, outc4, my4, opp4, outc3, my3, opp3, outc2, my2, opp2, outc1, my1, opp1, prob_r6):
    return get_optimal_move_r(stake, win_category, [outc5, outc4, outc3, outc2, outc1], [my5, my4, my3, my2, my1], [opp5, opp4, opp3, opp2, opp1], prob_r6)

def get_optimal_move_r7(stake, win_category, outc6, my6, opp6, outc5, my5, opp5, outc4, my4, opp4, outc3, my3, opp3, outc2, my2, opp2, outc1, my1, opp1, prob_r7):
    return get_optimal_move_r(stake, win_category, [outc6, outc5, outc4, outc3, outc2, outc1], [my6, my5, my4, my3, my2, my1], [opp6, opp5, opp4, opp3, opp2, opp1], prob_r7)

# -------------------- Работа с CSV --------------------
def load_data():
    numeric_cols = ['match_id', 'round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                    'score_me_before', 'score_opp_before', 'score_diff', 'streak_draws', 'is_last_round']
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
        # Приведение типов
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'player_name' in df.columns:
            df['player_name'] = df['player_name'].astype(str).replace('nan', '')
        else:
            df['player_name'] = ""
        # Добавление недостающих колонок
        for col in EXPECTED_COLS:
            if col not in df.columns:
                if col == 'score_diff':
                    df[col] = 0
                elif col == 'player_name':
                    df[col] = ""
                elif col == 'win_category':
                    continue
                elif col in numeric_cols:
                    df[col] = 0
                elif 'outcome' in col:
                    df[col] = 'none'
                else:
                    df[col] = '-1'
        if 'win_category' not in df.columns or df['win_category'].isna().any():
            df['win_category'] = df['opp_match_wins'].apply(compute_win_category)
        df['score_diff'] = df['score_me_before'] - df['score_opp_before']
        # Оставляем только нужные колонки в правильном порядке
        existing_cols = [c for c in EXPECTED_COLS if c in df.columns]
        df = df[existing_cols]
        return df
    except Exception as e:
        st.warning(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def ensure_csv():
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
    else:
        df = load_data()
        df.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')

def clean_unfinished():
    df = load_data()
    if df.empty:
        return
    finished_matches = set()
    for mid in df['match_id'].unique():
        match = df[df['match_id'] == mid].sort_values('round')
        for _, row in match.iterrows():
            score_me = row['score_me_before']
            score_opp = row['score_opp_before']
            outcome = row['outcome']
            if (outcome == 'win' and score_me + 1 >= 3) or (outcome == 'lose' and score_opp + 1 >= 3):
                finished_matches.add(mid)
                break
    df_clean = df[df['match_id'].isin(finished_matches)]
    if len(df_clean) != len(df):
        df_clean.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
        st.cache_data.clear()

def next_match_id():
    df = load_data()
    if df.empty:
        return 1
    return int(df['match_id'].max()) + 1

def get_last_n_records(n=10):
    df = load_data()
    if df.empty:
        return df
    df_last = df.tail(n).copy()
    # Преобразование для отображения
    for col in df_last.columns:
        if col.endswith('_move') and col not in ['my_move', 'opp_move']:
            df_last[col] = df_last[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
    if 'opp_move' in df_last.columns:
        df_last['opp_move'] = df_last['opp_move'].map(lambda x: LETTER_TO_MOVE.get(x, x))
    if 'my_move' in df_last.columns:
        df_last['my_move'] = df_last['my_move'].map(lambda x: LETTER_TO_MOVE.get(x, x))
    if 'outcome' in df_last.columns:
        df_last['outcome'] = df_last['outcome'].map(EN_TO_OUTCOME)
    for i in range(1,7):
        col = f'prev{i}_outcome'
        if col in df_last.columns:
            df_last[col] = df_last[col].map(lambda x: EN_TO_OUTCOME.get(x, x) if x != 'none' else 'нет')
    return df_last

# -------------------- Инициализация сессии --------------------
def init_session():
    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'setup'
        st.session_state.match_id = None
        st.session_state.round_num = 1
        st.session_state.history = []
        st.session_state.score_me = 0
        st.session_state.score_opp = 0
        st.session_state.streak_draws = 0
        st.session_state.opp_stats = {'wins': 0, 'winrate': 0.5, 'stake': 25}
        st.session_state.next_prediction = None
        st.session_state.selected_opp = None
        st.session_state.selected_outcome = None
        st.session_state.player_name = ""
        df_full = load_data()
        st.session_state.df_full = df_full
        st.session_state.prob_r2 = prepare_prob_r(df_full, 2)
        st.session_state.prob_r3 = prepare_prob_r(df_full, 3)
        st.session_state.prob_r4 = prepare_prob_r(df_full, 4)
        st.session_state.prob_r5 = prepare_prob_r(df_full, 5)
        st.session_state.prob_r6 = prepare_prob_r(df_full, 6)
        st.session_state.prob_r7 = prepare_prob_r(df_full, 7)
    else:
        df_full = load_data()
        st.session_state.df_full = df_full
        st.session_state.prob_r2 = prepare_prob_r(df_full, 2)
        st.session_state.prob_r3 = prepare_prob_r(df_full, 3)
        st.session_state.prob_r4 = prepare_prob_r(df_full, 4)
        st.session_state.prob_r5 = prepare_prob_r(df_full, 5)
        st.session_state.prob_r6 = prepare_prob_r(df_full, 6)
        st.session_state.prob_r7 = prepare_prob_r(df_full, 7)

# ========================
# Основное приложение
# ========================
st.set_page_config(page_title="Помощник в игре Камень - Ножницы - Бумага", layout="wide")
st.title("🎮 Помощник в игре 'Камень - Ножницы - Бумага'")

ensure_csv()
if 'clean_done' not in st.session_state:
    clean_unfinished()
    st.session_state.clean_done = True
init_session()

# -------------------- Начало матча --------------------
if st.session_state.game_state == 'setup':
    st.subheader("Новый матч")
    with st.form("setup"):
        col1, col2, col3 = st.columns(3)
        with col1:
            player_name = st.text_input("Имя противника", value="", placeholder="Например, Радушный Спасатель")
        with col2:
            wins = st.number_input("Побед противника (матчи)", min_value=-1, step=1, value=0)
        with col3:
            winrate_percent = st.number_input("Винрейт противника (%)", min_value=-100.0, max_value=100.0, step=0.01, value=50.0)
            winrate = winrate_percent / 100.0
        stake = st.selectbox("Ставка", [25, 50, 100])
        if st.form_submit_button("Начать матч"):
            st.session_state.player_name = player_name if player_name else "Неизвестный"
            st.session_state.opp_stats = {'wins': wins, 'winrate': winrate, 'stake': stake}
            st.session_state.match_id = next_match_id()
            st.session_state.game_state = 'playing'
            st.session_state.round_num = 1
            st.session_state.history = []
            st.session_state.score_me = 0
            st.session_state.score_opp = 0
            st.session_state.streak_draws = 0
            st.session_state.selected_opp = None
            st.session_state.selected_outcome = None
            win_cat = compute_win_category(wins)
            optimal, conf, sup = get_optimal_move_r1(stake, win_cat, st.session_state.df_full)
            st.session_state.next_prediction = (LETTER_TO_MOVE[optimal], LETTER_TO_MOVE[optimal], conf, sup)
            st.rerun()

# -------------------- Игровой процесс --------------------
elif st.session_state.game_state == 'playing':
    st.info(f"Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id} | Противник: {st.session_state.player_name}")

    if st.session_state.next_prediction:
        your_move, probable_opp, confidence, support = st.session_state.next_prediction
        if st.session_state.round_num <= 3:
            msg = f"🤖 Ваш оптимальный ход на **раунд {st.session_state.round_num}**: **{your_move}**\n\n📊 Наиболее вероятный ход противника: **{probable_opp}**"
        else:
            msg = f"🤖 Предсказание на **раунд {st.session_state.round_num}**: рекомендуемый ход – **{your_move}**"
        if support > 0:
            msg += f"\n\n📈 Основано на **{support}** примерах (уверенность {confidence:.1%})"
        else:
            msg += "\n\n⚠️ Недостаточно данных для статистики, рекомендован базовый ход."
        st.success(msg)

    st.subheader("Выберите ход противника и исход раунда")

    col1, col2, col3 = st.columns(3)
    opp_type_n = "primary" if st.session_state.selected_opp == "Ножницы" else "secondary"
    opp_type_k = "primary" if st.session_state.selected_opp == "Камень" else "secondary"
    opp_type_b = "primary" if st.session_state.selected_opp == "Бумага" else "secondary"

    with col1:
        if st.button("✌️ Ножницы", key="opp_n", use_container_width=True, type=opp_type_n):
            st.session_state.selected_opp = "Ножницы"
            st.rerun()
    with col2:
        if st.button("✊ Камень", key="opp_k", use_container_width=True, type=opp_type_k):
            st.session_state.selected_opp = "Камень"
            st.rerun()
    with col3:
        if st.button("✋ Бумага", key="opp_b", use_container_width=True, type=opp_type_b):
            st.session_state.selected_opp = "Бумага"
            st.rerun()

    st.markdown("**Исход для вас:**")
    col4, col5, col6 = st.columns(3)
    out_type_l = "primary" if st.session_state.selected_outcome == "Поражение" else "secondary"
    out_type_d = "primary" if st.session_state.selected_outcome == "Ничья" else "secondary"
    out_type_w = "primary" if st.session_state.selected_outcome == "Победа" else "secondary"

    with col4:
        if st.button("😞 Поражение", key="out_l", use_container_width=True, type=out_type_l):
            st.session_state.selected_outcome = "Поражение"
            st.rerun()
    with col5:
        if st.button("🤝 Ничья", key="out_d", use_container_width=True, type=out_type_d):
            st.session_state.selected_outcome = "Ничья"
            st.rerun()
    with col6:
        if st.button("😊 Победа", key="out_w", use_container_width=True, type=out_type_w):
            st.session_state.selected_outcome = "Победа"
            st.rerun()

    next_round_disabled = (st.session_state.selected_opp is None or st.session_state.selected_outcome is None)
    if st.button("➡️ Записать раунд и предсказать следующий", use_container_width=True, disabled=next_round_disabled):
        opp_move_full = st.session_state.selected_opp
        outcome_ru = st.session_state.selected_outcome
        opp_letter = MOVE_TO_LETTER[opp_move_full]
        outcome = OUTCOME_TO_EN[outcome_ru]

        beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
        lose = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
        if outcome == 'win':
            my_letter = beat[opp_letter]
        elif outcome == 'lose':
            my_letter = lose[opp_letter]
        else:
            my_letter = opp_letter

        # Заполняем prev1..prev6 из истории
        prev_data = {}
        for i in range(1, 7):
            if len(st.session_state.history) >= i:
                rec = st.session_state.history[-i]
                prev_data[f'prev{i}_opp_move'] = rec['opp_move']
                prev_data[f'prev{i}_my_move'] = rec['my_move']
                prev_data[f'prev{i}_outcome'] = rec['outcome']
            else:
                prev_data[f'prev{i}_opp_move'] = '-1'
                prev_data[f'prev{i}_my_move'] = '-1'
                prev_data[f'prev{i}_outcome'] = 'none'

        win_cat = compute_win_category(st.session_state.opp_stats['wins'])
        is_last_round = 1 if (st.session_state.score_me == 2 or st.session_state.score_opp == 2) else 0

        new_row = {
            'match_id': st.session_state.match_id,
            'round': st.session_state.round_num,
            'player_name': st.session_state.player_name,
            'win_category': win_cat,
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate'],
            'stake': st.session_state.opp_stats['stake'],
            'opp_move': opp_letter,
            'my_move': my_letter,
            'outcome': outcome,
            'score_me_before': st.session_state.score_me,
            'score_opp_before': st.session_state.score_opp,
            'score_diff': st.session_state.score_me - st.session_state.score_opp,
            'streak_draws': st.session_state.streak_draws,
            **prev_data,
            'is_last_round': is_last_round
        }

        st.session_state.history.append(new_row)

        # Обновляем счёт и streak
        if outcome == 'win':
            st.session_state.score_me += 1
            st.session_state.streak_draws = 0
        elif outcome == 'lose':
            st.session_state.score_opp += 1
            st.session_state.streak_draws = 0
        else:
            st.session_state.streak_draws += 1

        # Сохраняем в CSV
        df_new = pd.DataFrame([new_row])
        if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
            df_new.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
        else:
            df_new.to_csv(DATA_PATH, mode='a', header=False, index=False, sep=',', encoding='utf-8')
        st.cache_data.clear()
        time.sleep(0.1)

        # Проверка окончания матча
        if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
            st.session_state.game_state = 'finished'
            st.session_state.next_prediction = None
            clean_unfinished()
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        # --- ПРЕДСКАЗАНИЕ СЛЕДУЮЩЕГО РАУНДА ---
        next_round_num = st.session_state.round_num + 1
        stake_val = st.session_state.opp_stats['stake']
        win_cat = compute_win_category(st.session_state.opp_stats['wins'])

        def fallback_to_r1():
            opt, conf, sup = get_optimal_move_r1(stake_val, win_cat, st.session_state.df_full)
            return opt, conf, sup

        if next_round_num == 1:
            opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        elif next_round_num == 2:
            r1 = st.session_state.history[0]
            opt, conf, sup = get_optimal_move_r2(stake_val, win_cat, r1['outcome'], r1['my_move'], r1['opp_move'], st.session_state.prob_r2, None)
            if opt is None:
                opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        elif next_round_num == 3:
            r1, r2 = st.session_state.history[0], st.session_state.history[1]
            opt, conf, sup = get_optimal_move_r3(stake_val, win_cat, r2['outcome'], r2['my_move'], r2['opp_move'], r1['outcome'], r1['my_move'], r1['opp_move'], st.session_state.prob_r3, None)
            if opt is None:
                opt, conf, sup = get_optimal_move_r2(stake_val, win_cat, r2['outcome'], r2['my_move'], r2['opp_move'], st.session_state.prob_r2, None)
                if opt is None:
                    opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        elif next_round_num == 4:
            r1, r2, r3 = st.session_state.history[0], st.session_state.history[1], st.session_state.history[2]
            opt, conf, sup = get_optimal_move_r4(stake_val, win_cat,
                                                 r3['outcome'], r3['my_move'], r3['opp_move'],
                                                 r2['outcome'], r2['my_move'], r2['opp_move'],
                                                 r1['outcome'], r1['my_move'], r1['opp_move'],
                                                 st.session_state.prob_r4)
            if opt is None:
                opt, conf, sup = get_optimal_move_r3(stake_val, win_cat,
                                                     r3['outcome'], r3['my_move'], r3['opp_move'],
                                                     r2['outcome'], r2['my_move'], r2['opp_move'],
                                                     st.session_state.prob_r3, None)
                if opt is None:
                    opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        elif next_round_num == 5:
            r1, r2, r3, r4 = st.session_state.history[0], st.session_state.history[1], st.session_state.history[2], st.session_state.history[3]
            opt, conf, sup = get_optimal_move_r5(stake_val, win_cat,
                                                 r4['outcome'], r4['my_move'], r4['opp_move'],
                                                 r3['outcome'], r3['my_move'], r3['opp_move'],
                                                 r2['outcome'], r2['my_move'], r2['opp_move'],
                                                 r1['outcome'], r1['my_move'], r1['opp_move'],
                                                 st.session_state.prob_r5)
            if opt is None:
                opt, conf, sup = get_optimal_move_r4(stake_val, win_cat,
                                                     r4['outcome'], r4['my_move'], r4['opp_move'],
                                                     r3['outcome'], r3['my_move'], r3['opp_move'],
                                                     r2['outcome'], r2['my_move'], r2['opp_move'],
                                                     st.session_state.prob_r4)
                if opt is None:
                    opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        elif next_round_num == 6:
            r1, r2, r3, r4, r5 = st.session_state.history[0], st.session_state.history[1], st.session_state.history[2], st.session_state.history[3], st.session_state.history[4]
            opt, conf, sup = get_optimal_move_r6(stake_val, win_cat,
                                                 r5['outcome'], r5['my_move'], r5['opp_move'],
                                                 r4['outcome'], r4['my_move'], r4['opp_move'],
                                                 r3['outcome'], r3['my_move'], r3['opp_move'],
                                                 r2['outcome'], r2['my_move'], r2['opp_move'],
                                                 r1['outcome'], r1['my_move'], r1['opp_move'],
                                                 st.session_state.prob_r6)
            if opt is None:
                opt, conf, sup = get_optimal_move_r5(stake_val, win_cat,
                                                     r5['outcome'], r5['my_move'], r5['opp_move'],
                                                     r4['outcome'], r4['my_move'], r4['opp_move'],
                                                     r3['outcome'], r3['my_move'], r3['opp_move'],
                                                     r2['outcome'], r2['my_move'], r2['opp_move'],
                                                     st.session_state.prob_r5)
                if opt is None:
                    opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        elif next_round_num == 7:
            r1, r2, r3, r4, r5, r6 = st.session_state.history[0], st.session_state.history[1], st.session_state.history[2], st.session_state.history[3], st.session_state.history[4], st.session_state.history[5]
            opt, conf, sup = get_optimal_move_r7(stake_val, win_cat,
                                                 r6['outcome'], r6['my_move'], r6['opp_move'],
                                                 r5['outcome'], r5['my_move'], r5['opp_move'],
                                                 r4['outcome'], r4['my_move'], r4['opp_move'],
                                                 r3['outcome'], r3['my_move'], r3['opp_move'],
                                                 r2['outcome'], r2['my_move'], r2['opp_move'],
                                                 r1['outcome'], r1['my_move'], r1['opp_move'],
                                                 st.session_state.prob_r7)
            if opt is None:
                opt, conf, sup = get_optimal_move_r6(stake_val, win_cat,
                                                     r6['outcome'], r6['my_move'], r6['opp_move'],
                                                     r5['outcome'], r5['my_move'], r5['opp_move'],
                                                     r4['outcome'], r4['my_move'], r4['opp_move'],
                                                     r3['outcome'], r3['my_move'], r3['opp_move'],
                                                     r2['outcome'], r2['my_move'], r2['opp_move'],
                                                     st.session_state.prob_r6)
                if opt is None:
                    opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)
        else:
            opt, conf, sup = fallback_to_r1()
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[opt], conf, sup)

        st.session_state.round_num = next_round_num
        st.session_state.selected_opp = None
        st.session_state.selected_outcome = None
        st.rerun()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("---")
        st.subheader("📊 Ходы противника в текущем матче")
        if st.session_state.history:
            moves = []
            for rec in st.session_state.history:
                move_name = LETTER_TO_MOVE.get(rec['opp_move'], "?")
                emoji = MOVE_EMOJI.get(move_name, "❓")
                moves.append(f"{emoji} {move_name}")
            st.write(" → ".join(moves))
        else:
            st.write("Пока нет записанных ходов.")
    with col_right:
        st.markdown("---")
        st.subheader("📋 Последние 10 сохранённых раундов")
        last_records = get_last_n_records(10)
        if not last_records.empty:
            show_cols = ['match_id', 'round', 'player_name', 'win_category', 'opp_move', 'my_move', 'outcome', 'score_me_before', 'score_opp_before']
            available = [c for c in show_cols if c in last_records.columns]
            st.dataframe(last_records[available], use_container_width=True, height=400)
            if os.path.exists(DATA_PATH):
                with open(DATA_PATH, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
                st.download_button("💾 Скачать CSV", data=csv_data, file_name="rps_data.csv", mime="text/csv")
        else:
            st.write("Нет записей. После первого раунда данные появятся.")

# -------------------- Завершение матча --------------------
elif st.session_state.game_state == 'finished':
    st.info(f"Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp} | Противник: {st.session_state.player_name}")
    if st.button("➕ Начать новый матч", use_container_width=True):
        st.session_state.game_state = 'setup'
        st.session_state.history = []
        st.session_state.score_me = 0
        st.session_state.score_opp = 0
        st.session_state.round_num = 1
        st.session_state.streak_draws = 0
        st.session_state.next_prediction = None
        st.session_state.selected_opp = None
        st.session_state.selected_outcome = None
        st.rerun()