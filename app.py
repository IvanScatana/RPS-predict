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

# Базовые колонки (без prev)
BASE_COLS = [
    'match_id', 'round', 'player_name', 'win_category',
    'opp_match_wins', 'opp_match_winrate', 'stake',
    'opp_move', 'my_move', 'outcome',
    'score_me_before', 'score_opp_before', 'score_diff', 'streak_draws'
]

# Генерируем все prev-колонки для сдвигов 1..6
PREV_COLS = []
for shift in range(1, 7):
    PREV_COLS.extend([f'prev{shift}_opp_move', f'prev{shift}_my_move', f'prev{shift}_outcome'])

# Порядок колонок: базовые, затем is_last_round, затем prev*
EXPECTED_COLS = BASE_COLS + ['is_last_round'] + PREV_COLS

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

# -------------------- Загрузка и подготовка данных (с кешем) --------------------
@st.cache_data(ttl=3600)
def load_data_cached():
    numeric_cols = ['match_id', 'round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                    'score_me_before', 'score_opp_before', 'score_diff', 'streak_draws', 'is_last_round']
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
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
        df = df[[c for c in EXPECTED_COLS if c in df.columns]]
        return df
    except Exception as e:
        st.warning(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def prepare_prob_table(df, round_num):
    """Строит таблицу вероятностей для заданного раунда (только точное совпадение)."""
    df_r = df[df['round'] == round_num].copy()
    if df_r.empty:
        return pd.DataFrame()
    group_cols = ['stake', 'win_category']
    for i in range(1, round_num):
        group_cols.append(f'prev{i}_outcome')
        group_cols.append(f'prev{i}_my_move')
        group_cols.append(f'prev{i}_opp_move')
    counts = df_r.groupby(group_cols)['opp_move'].value_counts().reset_index(name='count')
    counts['prob'] = counts.groupby(group_cols)['count'].transform(lambda x: x / x.sum())
    return counts

# -------------------- Стратегия первого раунда (базовая) --------------------
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
    # Уверенность = вероятность не проиграть (победа + ничья)
    win_condition = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
    win_count = counts.get(win_condition[best], 0)
    draw_count = counts.get(best, 0)
    confidence = (win_count + draw_count) / total if total > 0 else 0
    return best, confidence, total

def get_most_probable_opp_r1(stake, win_category, df):
    sub = df[(df["round"] == 1) & (df["stake"] == stake) & (df["win_category"] == win_category)]
    if len(sub) == 0:
        sub = df[(df["round"] == 1) & (df["win_category"] == win_category)]
        if len(sub) == 0:
            sub = df[(df["round"] == 1) & (df["stake"] == stake)]
            if len(sub) == 0:
                return 'К'
    return sub['opp_move'].mode()[0] if not sub.empty else 'К'

# -------------------- Обобщённая функция для раундов 2..7 (точное совпадение) --------------------
def get_move_for_round(round_num, stake, win_category, history, prob_table, df_full):
    if prob_table.empty:
        return None, 0, 0, None
    mask = (prob_table['stake'] == stake) & (prob_table['win_category'] == win_category)
    for i in range(1, round_num):
        col_outc = f'prev{i}_outcome'
        col_my = f'prev{i}_my_move'
        col_opp = f'prev{i}_opp_move'
        if col_outc not in prob_table.columns:
            return None, 0, 0, None
        rec = history[-i]
        mask &= (prob_table[col_outc] == rec['outcome'])
        mask &= (prob_table[col_my] == rec['my_move'])
        mask &= (prob_table[col_opp] == rec['opp_move'])
    subset = prob_table[mask]
    if len(subset) == 0:
        return None, 0, 0, None
    total = subset['count'].sum()
    # Вычисляем для каждого своего хода вероятность не проиграть
    not_lose = {}
    for my_move in ['К','Н','Б']:
        win_draw = 0
        for opp, cnt in zip(subset['opp_move'], subset['count']):
            if my_move == opp:
                win_draw += cnt
            elif (my_move == 'К' and opp == 'Н') or (my_move == 'Н' and opp == 'Б') or (my_move == 'Б' and opp == 'К'):
                win_draw += cnt
        not_lose[my_move] = win_draw / total
    best_move = max(not_lose, key=not_lose.get)
    confidence = not_lose[best_move]
    support = total
    opp_counts = subset.groupby('opp_move')['count'].sum()
    most_probable_opp = opp_counts.idxmax() if not opp_counts.empty else 'К'
    return best_move, confidence, support, most_probable_opp

# -------------------- Обёртки для конкретных раундов --------------------
def get_optimal_move_r2(stake, win_category, history, prob_r2, df_full):
    return get_move_for_round(2, stake, win_category, history, prob_r2, df_full)

def get_optimal_move_r3(stake, win_category, history, prob_r3, df_full):
    return get_move_for_round(3, stake, win_category, history, prob_r3, df_full)

def get_optimal_move_r4(stake, win_category, history, prob_r4, df_full):
    return get_move_for_round(4, stake, win_category, history, prob_r4, df_full)

def get_optimal_move_r5(stake, win_category, history, prob_r5, df_full):
    return get_move_for_round(5, stake, win_category, history, prob_r5, df_full)

def get_optimal_move_r6(stake, win_category, history, prob_r6, df_full):
    return get_move_for_round(6, stake, win_category, history, prob_r6, df_full)

def get_optimal_move_r7(stake, win_category, history, prob_r7, df_full):
    return get_move_for_round(7, stake, win_category, history, prob_r7, df_full)

# -------------------- Работа с CSV --------------------
def ensure_csv():
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
    else:
        df = load_data_cached()
        df.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')

def clean_unfinished():
    df = load_data_cached()
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
    df = load_data_cached()
    if df.empty:
        return 1
    return int(df['match_id'].max()) + 1

def get_last_n_records(n=10):
    df = load_data_cached()
    if df.empty:
        return df
    df_last = df.tail(n).copy()
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
    if 'df_full' not in st.session_state:
        st.session_state.df_full = load_data_cached()
        st.session_state.prob_r2 = prepare_prob_table(st.session_state.df_full, 2)
        st.session_state.prob_r3 = prepare_prob_table(st.session_state.df_full, 3)
        st.session_state.prob_r4 = prepare_prob_table(st.session_state.df_full, 4)
        st.session_state.prob_r5 = prepare_prob_table(st.session_state.df_full, 5)
        st.session_state.prob_r6 = prepare_prob_table(st.session_state.df_full, 6)
        st.session_state.prob_r7 = prepare_prob_table(st.session_state.df_full, 7)

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
            st.session_state.player_name = player_name if player_name else "Неизвестен"
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
            probable_opp = get_most_probable_opp_r1(stake, win_cat, st.session_state.df_full)
            # Проверка корректности
            if optimal not in LETTER_TO_MOVE:
                optimal = 'К'
            if probable_opp not in LETTER_TO_MOVE:
                probable_opp = 'К'
            st.session_state.next_prediction = (LETTER_TO_MOVE[optimal], LETTER_TO_MOVE[probable_opp], conf, sup)
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
            msg += "\n\n⚠️ Нет статистики для данного контекста, рекомендован базовый ход."
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
            'is_last_round': is_last_round,
            **prev_data
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
        time.sleep(0.05)

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

        def base_prediction():
            opt, conf, sup = get_optimal_move_r1(stake_val, win_cat, st.session_state.df_full)
            prob_opp = get_most_probable_opp_r1(stake_val, win_cat, st.session_state.df_full)
            return opt, conf, sup, prob_opp

        if next_round_num == 1:
            opt, conf, sup, prob_opp = base_prediction()
        elif next_round_num == 2:
            res = get_optimal_move_r2(stake_val, win_cat, st.session_state.history, st.session_state.prob_r2, st.session_state.df_full)
            if res is None:
                opt, conf, sup, prob_opp = base_prediction()
            else:
                opt, conf, sup, prob_opp = res
        elif next_round_num == 3:
            res = get_optimal_move_r3(stake_val, win_cat, st.session_state.history, st.session_state.prob_r3, st.session_state.df_full)
            if res is None:
                opt, conf, sup, prob_opp = base_prediction()
            else:
                opt, conf, sup, prob_opp = res
        elif next_round_num == 4:
            res = get_optimal_move_r4(stake_val, win_cat, st.session_state.history, st.session_state.prob_r4, st.session_state.df_full)
            if res is None:
                opt, conf, sup, prob_opp = base_prediction()
            else:
                opt, conf, sup, prob_opp = res
        elif next_round_num == 5:
            res = get_optimal_move_r5(stake_val, win_cat, st.session_state.history, st.session_state.prob_r5, st.session_state.df_full)
            if res is None:
                opt, conf, sup, prob_opp = base_prediction()
            else:
                opt, conf, sup, prob_opp = res
        elif next_round_num == 6:
            res = get_optimal_move_r6(stake_val, win_cat, st.session_state.history, st.session_state.prob_r6, st.session_state.df_full)
            if res is None:
                opt, conf, sup, prob_opp = base_prediction()
            else:
                opt, conf, sup, prob_opp = res
        elif next_round_num == 7:
            res = get_optimal_move_r7(stake_val, win_cat, st.session_state.history, st.session_state.prob_r7, st.session_state.df_full)
            if res is None:
                opt, conf, sup, prob_opp = base_prediction()
            else:
                opt, conf, sup, prob_opp = res
        else:
            opt, conf, sup, prob_opp = base_prediction()

        # Проверка корректности перед использованием словаря
        if opt not in LETTER_TO_MOVE:
            opt = 'К'
        if prob_opp not in LETTER_TO_MOVE:
            prob_opp = 'К'
        st.session_state.next_prediction = (LETTER_TO_MOVE[opt], LETTER_TO_MOVE[prob_opp], conf, sup)
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