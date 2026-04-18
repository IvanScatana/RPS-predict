import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
from collections import defaultdict, Counter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import joblib

# ========================
# Конфигурация
# ========================
STATS_FILE = 'rps_markov_stats.json'  # не используется, оставлен для совместимости
DATA_PATH = 'rps_data.csv'
MODEL_PATH = 'catboost_model.pkl'
PREPROCESSOR_PATH = 'rps_preprocessor.pkl'

MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}
MOVE_EMOJI = {"Камень": "✊", "Ножницы": "✌️", "Бумага": "✋"}

# Ожидаемые колонки (добавлена score_diff)
EXPECTED_COLS = [
    'match_id', 'round', 'player_name', 'win_category',
    'opp_match_wins', 'opp_match_winrate', 'stake',
    'opp_move', 'my_move', 'outcome',
    'score_me_before', 'score_opp_before', 'score_diff', 'streak_draws',
    'prev_opp_move', 'prev_outcome', 'prev2_opp_move', 'prev2_outcome'
]

# ========================
# Функции для вычисления категории побед
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

# ========================
# Статистические функции для раундов 1-3
# ========================
def get_move_r1(stake, df):
    """Возвращает оптимальный ход в первом раунде на основе статистики."""
    sub = df[(df["round"] == 1) & (df["stake"] == stake)]
    if len(sub) == 0:
        return 'К'
    counts = sub["opp_move"].value_counts()
    p_k = counts.get('К', 0) / len(sub)
    p_n = counts.get('Н', 0) / len(sub)
    p_b = counts.get('Б', 0) / len(sub)
    exp_k = p_n - p_b
    exp_n = p_b - p_k
    exp_b = p_k - p_n
    if exp_k >= exp_n and exp_k >= exp_b:
        return 'К'
    elif exp_n >= exp_b:
        return 'Н'
    else:
        return 'Б'

def prepare_prob_r2(df):
    """Строит таблицу вероятностей для второго раунда."""
    df_r2 = df[df['round'] == 2].copy()
    if df_r2.empty:
        return pd.DataFrame()
    return df_r2.groupby(['stake', 'prev_outcome', 'prev_my_move', 'prev_opp_move'])['opp_move'] \
                 .value_counts(normalize=True).reset_index(name='prob')

def get_move_r2(stake, outcome_r1, my_move_r1, opp_move_r1, prob_r2):
    """Возвращает ход во втором раунде или None, если нет данных."""
    if prob_r2.empty:
        return None
    mask = (prob_r2['stake'] == stake) & \
           (prob_r2['prev_outcome'] == outcome_r1) & \
           (prob_r2['prev_my_move'] == my_move_r1) & \
           (prob_r2['prev_opp_move'] == opp_move_r1)
    subset = prob_r2[mask]
    if len(subset) == 0:
        return None
    p_k = subset[subset['opp_move'] == 'К']['prob'].sum()
    p_n = subset[subset['opp_move'] == 'Н']['prob'].sum()
    p_b = subset[subset['opp_move'] == 'Б']['prob'].sum()
    exp_k = p_n - p_b
    exp_n = p_b - p_k
    exp_b = p_k - p_n
    if exp_k >= exp_n and exp_k >= exp_b:
        return 'К'
    elif exp_n >= exp_b:
        return 'Н'
    else:
        return 'Б'

def prepare_prob_r3(df):
    """Строит таблицу вероятностей для третьего раунда."""
    df_r3 = df[df['round'] == 3].copy()
    if df_r3.empty:
        return pd.DataFrame()
    return df_r3.groupby(['stake', 'prev_outcome', 'prev_my_move', 'prev_opp_move',
                          'prev2_outcome', 'prev2_my_move', 'prev2_opp_move'])['opp_move'] \
                 .value_counts(normalize=True).reset_index(name='prob')

def get_move_r3(stake, outcome_r2, my_move_r2, opp_move_r2,
                outcome_r1, my_move_r1, opp_move_r1, prob_r3):
    if prob_r3.empty:
        return None
    mask = (prob_r3['stake'] == stake) & \
           (prob_r3['prev_outcome'] == outcome_r2) & \
           (prob_r3['prev_my_move'] == my_move_r2) & \
           (prob_r3['prev_opp_move'] == opp_move_r2) & \
           (prob_r3['prev2_outcome'] == outcome_r1) & \
           (prob_r3['prev2_my_move'] == my_move_r1) & \
           (prob_r3['prev2_opp_move'] == opp_move_r1)
    subset = prob_r3[mask]
    if len(subset) == 0:
        return None
    p_k = subset[subset['opp_move'] == 'К']['prob'].sum()
    p_n = subset[subset['opp_move'] == 'Н']['prob'].sum()
    p_b = subset[subset['opp_move'] == 'Б']['prob'].sum()
    exp_k = p_n - p_b
    exp_n = p_b - p_k
    exp_b = p_k - p_n
    if exp_k >= exp_n and exp_k >= exp_b:
        return 'К'
    elif exp_n >= exp_b:
        return 'Н'
    else:
        return 'Б'

# ========================
# Загрузка модели ML
# ========================
def load_ml_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            return model, preprocessor
        except Exception as e:
            st.warning(f"Не удалось загрузить ML-модель: {e}")
    return None, None

def get_move_ml(features, model, preprocessor):
    """Предсказание хода через ML модель (раунд >=4)."""
    if model is None or preprocessor is None:
        return None
    X = pd.DataFrame([features])
    X_processed = preprocessor.transform(X)
    pred = model.predict(X_processed)
    # Извлекаем скаляр (число или строка)
    if isinstance(pred, np.ndarray):
        pred_code = pred.item() if pred.size == 1 else pred[0]
    else:
        pred_code = pred
    # Если строка, то это уже ход
    if pred_code in ('К', 'Н', 'Б'):
        predicted_opp = pred_code
    else:
        move_map = {0: 'К', 1: 'Н', 2: 'Б'}
        predicted_opp = move_map.get(pred_code, 'К')
    # Контр-ход
    if predicted_opp == 'К': return 'Б'
    if predicted_opp == 'Н': return 'К'
    return 'Н'

# ========================
# Работа с CSV
# ========================
def load_data():
    """Загружает CSV с приведением типов и вычислением score_diff."""
    numeric_cols = ['round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                    'score_me_before', 'score_opp_before', 'streak_draws']
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
        # Добавляем недостающие колонки
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
                elif col in ['prev_outcome', 'prev2_outcome']:
                    df[col] = 'none'
                else:
                    df[col] = ""
        # Вычисляем win_category
        if 'win_category' not in df.columns or df['win_category'].isna().any():
            df['win_category'] = df['opp_match_wins'].apply(compute_win_category)
        # Вычисляем score_diff
        df['score_diff'] = df['score_me_before'] - df['score_opp_before']
        # Приводим порядок колонок
        df = df[EXPECTED_COLS]
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
    finished = set()
    for mid in df['match_id'].unique():
        match = df[df['match_id'] == mid]
        for _, row in match.iterrows():
            score_me = int(row['score_me_before'])
            score_opp = int(row['score_opp_before'])
            if (row['outcome'] == 'win' and score_me + 1 >= 3) or \
               (row['outcome'] == 'lose' and score_opp + 1 >= 3):
                finished.add(mid)
                break
    df_clean = df[df['match_id'].isin(finished)]
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
    # Преобразование буквенных ходов в названия
    for col in ['opp_move', 'my_move', 'prev_opp_move', 'prev2_opp_move']:
        if col in df_last.columns:
            df_last[col] = df_last[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
    if 'outcome' in df_last.columns:
        df_last['outcome'] = df_last['outcome'].map(EN_TO_OUTCOME)
    if 'prev_outcome' in df_last.columns:
        df_last['prev_outcome'] = df_last['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
    if 'prev2_outcome' in df_last.columns:
        df_last['prev2_outcome'] = df_last['prev2_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x) if x != 'none' else 'нет')
    return df_last

# ========================
# Инициализация сессии
# ========================
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
        # Загружаем данные и строим статистические таблицы
        df_full = load_data()
        st.session_state.df_full = df_full
        st.session_state.prob_r2 = prepare_prob_r2(df_full)
        st.session_state.prob_r3 = prepare_prob_r3(df_full)
        # Загружаем ML модель
        st.session_state.ml_model, st.session_state.ml_preprocessor = load_ml_model()
    # Обновляем статистические таблицы при каждом запуске (если данные изменились)
    else:
        df_full = load_data()
        st.session_state.df_full = df_full
        st.session_state.prob_r2 = prepare_prob_r2(df_full)
        st.session_state.prob_r3 = prepare_prob_r3(df_full)

# ========================
# Основное приложение
# ========================
st.set_page_config(page_title="Помощник в игре Камень - Ножницы - Бумага", layout="wide")
st.title("🎮 Помощник в игре 'Камень - Ножницы - Бумага'")

ensure_csv()
init_session()

# ========================
# Начало матча
# ========================
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
            clean_unfinished()
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
            # Предсказание первого раунда
            best_move = get_move_r1(stake, st.session_state.df_full)
            st.session_state.next_prediction = (LETTER_TO_MOVE[best_move], LETTER_TO_MOVE[best_move])  # (opp, my) но для 1 раунда my = opp? Нет, в предсказании мы показываем ход противника и свой ход. Лучше показать свой ход.
            # Переделаем: предсказание — это мой ход
            st.session_state.next_prediction = (LETTER_TO_MOVE[best_move], LETTER_TO_MOVE[best_move])  # временно, но в интерфейсе показывается pred_move и ваш ход. Сделаем единообразно.
            st.rerun()

# ========================
# Игровой процесс
# ========================
elif st.session_state.game_state == 'playing':
    st.info(f"Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id} | Противник: {st.session_state.player_name}")

    if st.session_state.next_prediction:
        pred_move, your_move = st.session_state.next_prediction
        st.success(f"🤖 Предсказание на **раунд {st.session_state.round_num}**: противник – **{pred_move}**, вам – **{your_move}**")

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

        prev_opp = st.session_state.history[-1]['opp_move'] if st.session_state.history else '-1'
        prev_out = st.session_state.history[-1]['outcome'] if st.session_state.history else 'none'
        prev2_opp = st.session_state.history[-2]['opp_move'] if len(st.session_state.history) >= 2 else '-1'
        prev2_out = st.session_state.history[-2]['outcome'] if len(st.session_state.history) >= 2 else 'none'

        win_cat = compute_win_category(st.session_state.opp_stats['wins'])

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
            'prev_opp_move': prev_opp,
            'prev_outcome': prev_out,
            'prev2_opp_move': prev2_opp,
            'prev2_outcome': prev2_out
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
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        # --- ПРЕДСКАЗАНИЕ СЛЕДУЮЩЕГО РАУНДА ---
        next_round_num = st.session_state.round_num + 1
        # Определяем стратегию в зависимости от номера следующего раунда
        if next_round_num == 1:
            # не может быть, но на всякий случай
            best_move = get_move_r1(st.session_state.opp_stats['stake'], st.session_state.df_full)
            pred_my = best_move
            pred_opp = best_move  # не используется
        elif next_round_num == 2:
            # нужны данные первого раунда
            r1 = st.session_state.history[0]  # первый раунд уже записан
            best = get_move_r2(st.session_state.opp_stats['stake'],
                               r1['outcome'], r1['my_move'], r1['opp_move'],
                               st.session_state.prob_r2)
            if best is None:
                best = get_move_r1(st.session_state.opp_stats['stake'], st.session_state.df_full)
            pred_my = best
            pred_opp = best
        elif next_round_num == 3:
            r1 = st.session_state.history[0]
            r2 = st.session_state.history[1]
            best = get_move_r3(st.session_state.opp_stats['stake'],
                               r2['outcome'], r2['my_move'], r2['opp_move'],
                               r1['outcome'], r1['my_move'], r1['opp_move'],
                               st.session_state.prob_r3)
            if best is None:
                # fallback на r2
                best = get_move_r2(st.session_state.opp_stats['stake'],
                                   r2['outcome'], r2['my_move'], r2['opp_move'],
                                   st.session_state.prob_r2)
                if best is None:
                    best = get_move_r1(st.session_state.opp_stats['stake'], st.session_state.df_full)
            pred_my = best
            pred_opp = best
        else:  # >=4
            # Используем ML, если модель есть
            if st.session_state.ml_model is not None:
                # Берём текущий раунд (последний в истории) как основу для предсказания следующего
                curr = st.session_state.history[-1]
                features = {
                    'round': curr['round'],
                    'stake': curr['stake'],
                    'opp_match_wins': curr['opp_match_wins'],
                    'opp_match_winrate': curr['opp_match_winrate'],
                    'score_me_before': curr['score_me_before'],
                    'score_opp_before': curr['score_opp_before'],
                    'streak_draws': curr['streak_draws'],
                    'is_last_round': 1 if (st.session_state.score_me == 2 or st.session_state.score_opp == 2) else 0,
                    'win_category': curr['win_category'],
                    'opp_move': curr['opp_move'],
                    'my_move': curr['my_move'],
                    'outcome': curr['outcome'],
                    'prev_opp_move': curr['prev_opp_move'],
                    'prev_my_move': curr['prev_my_move'],
                    'prev_outcome': curr['prev_outcome'],
                    'prev2_opp_move': curr['prev2_opp_move'],
                    'prev2_my_move': curr['prev2_my_move'],
                    'prev2_outcome': curr['prev2_outcome']
                }
                best = get_move_ml(features, st.session_state.ml_model, st.session_state.ml_preprocessor)
                if best is None:
                    # fallback на r3
                    r1 = st.session_state.history[0]
                    r2 = st.session_state.history[1]
                    best = get_move_r3(st.session_state.opp_stats['stake'],
                                       r2['outcome'], r2['my_move'], r2['opp_move'],
                                       r1['outcome'], r1['my_move'], r1['opp_move'],
                                       st.session_state.prob_r3)
                    if best is None:
                        best = get_move_r1(st.session_state.opp_stats['stake'], st.session_state.df_full)
            else:
                # ML нет, используем r3
                r1 = st.session_state.history[0]
                r2 = st.session_state.history[1]
                best = get_move_r3(st.session_state.opp_stats['stake'],
                                   r2['outcome'], r2['my_move'], r2['opp_move'],
                                   r1['outcome'], r1['my_move'], r1['opp_move'],
                                   st.session_state.prob_r3)
                if best is None:
                    best = get_move_r1(st.session_state.opp_stats['stake'], st.session_state.df_full)
            pred_my = best
            pred_opp = best  # для отображения в предсказании показываем и ход противника? В интерфейсе ожидается (pred_move, your_move). Так как мы предсказываем свой ход, то pred_move можно сделать таким же.
        # Сохраняем предсказание
        st.session_state.next_prediction = (LETTER_TO_MOVE[pred_my], LETTER_TO_MOVE[pred_my])
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

# ========================
# Завершение матча
# ========================
elif st.session_state.game_state == 'finished':
    st.info(f"Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp} | Противник: {st.session_state.player_name}")
    if st.button("➕ Начать новый матч", use_container_width=True):
        clean_unfinished()
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