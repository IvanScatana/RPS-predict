import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime

# ========================
# Конфигурация
# ========================
MODEL_PATH = 'rps_model.pkl'
TARGET_ENCODER_PATH = 'target_encoder.pkl'
DATA_PATH = 'rps_data.csv'

# ========================
# Загрузка модели и кодировщика
# ========================
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load(MODEL_PATH)
        le_target = joblib.load(TARGET_ENCODER_PATH)
        return pipeline, le_target
    except:
        st.error("Модель не найдена. Сначала обучите модель и сохраните файлы.")
        return None, None

# ========================
# Функция для получения признаков текущего раунда из истории
# ========================
def get_features_for_round(history, current_round_index):
    """
    history: список словарей с раундами (каждый содержит opp_move, outcome, my_move, score_me_before, ...)
    current_round_index: индекс текущего раунда (который уже сыгран, для него известны opp_move и outcome)
    Возвращает словарь признаков для предсказания следующего раунда.
    """
    if not history or current_round_index < 0:
        return None
    
    current = history[current_round_index]
    prev = history[current_round_index - 1] if current_round_index > 0 else None
    prev2 = history[current_round_index - 2] if current_round_index > 1 else None
    
    # Признаки для предсказания следующего хода
    features = {
        'opp_move': current['opp_move'],
        'my_move': current['my_move'],
        'outcome': current['outcome'],
        'prev_opp_move': prev['opp_move'] if prev else '-1',
        'prev_outcome': prev['outcome'] if prev else 'none',
        'prev2_opp_move': prev2['opp_move'] if prev2 else '-1',
        'score_me_before': current['score_me_before'],
        'score_opp_before': current['score_opp_before'],
        'streak_draws': current['streak_draws'],
        'stake': current['stake'],
        'opp_match_wins': current['opp_match_wins'],
        'opp_match_winrate': current['opp_match_winrate'],
    }
    return features

# ========================
# Функция предсказания
# ========================
def predict_next_move(pipeline, le_target, features_dict):
    input_df = pd.DataFrame([features_dict])
    pred_enc = pipeline.predict(input_df)[0]
    pred_move = le_target.inverse_transform([pred_enc])[0]
    beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
    your_move = beat[pred_move]
    return pred_move, your_move

# ========================
# Функция для вычисления счёта и streak_draws
# ========================
def update_score_and_streak(history, new_outcome):
    """
    На основе предыдущей истории и нового исхода вычисляет новый счёт и streak_draws
    """
    if not history:
        score_me = 0
        score_opp = 0
        streak = 0
    else:
        last = history[-1]
        score_me = last['score_me_before']
        score_opp = last['score_opp_before']
        streak = last['streak_draws']
        if last['outcome'] == 'draw':
            streak += 1
        else:
            streak = 0
        # Обновляем счёт
        if new_outcome == 'win':
            score_me += 1
        elif new_outcome == 'lose':
            score_opp += 1
    return score_me, score_opp, streak

# ========================
# Инициализация сессии
# ========================
if 'match_active' not in st.session_state:
    st.session_state.match_active = False
    st.session_state.match_id = 1
    st.session_state.round_num = 1
    st.session_state.history = []
    st.session_state.match_stats = {}  # opp_match_wins, opp_match_winrate, stake
    st.session_state.prediction = None

# ========================
# Заголовок
# ========================
st.set_page_config(page_title="RPS Predictor", layout="centered")
st.title("🎮 Предсказатель хода в 'Камень-Ножницы-Бумага'")
st.markdown("Введите минимальные данные, модель предскажет следующий ход противника.")

# Загрузка модели
pipeline, le_target = load_model()
if pipeline is None:
    st.stop()

# ========================
# Режим: начало нового матча
# ========================
if not st.session_state.match_active:
    st.subheader("Новый матч")
    with st.form("new_match"):
        col1, col2, col3 = st.columns(3)
        with col1:
            opp_match_wins = st.number_input("Побед противника в матчах (ист.)", min_value=-1, step=1, value=0)
        with col2:
            opp_match_winrate = st.number_input("Винрейт противника (0..1)", min_value=-1.0, max_value=1.0, step=0.05, value=0.5)
        with col3:
            stake = st.selectbox("Ставка", [25, 50, 100], index=0)
        start = st.form_submit_button("Начать матч")
        if start:
            st.session_state.match_active = True
            st.session_state.match_id = 1  # можно автоинкремент, но для простоты 1
            st.session_state.round_num = 1
            st.session_state.history = []
            st.session_state.match_stats = {
                'opp_match_wins': opp_match_wins,
                'opp_match_winrate': opp_match_winrate,
                'stake': stake
            }
            st.session_state.prediction = None
            st.rerun()

# ========================
# Режим: активный матч
# ========================
if st.session_state.match_active:
    # Отображаем текущий счёт
    if st.session_state.history:
        last = st.session_state.history[-1]
        score_me = last['score_me_before'] + (1 if last['outcome'] == 'win' else 0)
        score_opp = last['score_opp_before'] + (1 if last['outcome'] == 'lose' else 0)
        st.info(f"📊 Счёт: **{score_me} : {score_opp}**")
    else:
        st.info("📊 Счёт: 0 : 0")

    # Если матч ещё не закончен (у кого-то меньше 3 побед)
    if not st.session_state.history or (score_me < 3 and score_opp < 3):
        # Форма для ввода текущего раунда
        with st.form("round_form"):
            st.subheader(f"Раунд {st.session_state.round_num}")
            col1, col2 = st.columns(2)
            with col1:
                opp_move = st.selectbox("Ход противника", ["К", "Н", "Б"], key="opp")
            with col2:
                outcome = st.selectbox("Исход для вас", ["win", "lose", "draw"], key="out")
            submitted = st.form_submit_button("Записать раунд и предсказать следующий")

        # Если нет истории, предсказать первый ход (без предыдущих данных)
        if not st.session_state.history and not submitted:
            # Предсказание первого хода (без контекста) – используем заглушки
            features = {
                'opp_move': 'К',  # заглушка
                'my_move': 'К',   # заглушка
                'outcome': 'draw', # заглушка
                'prev_opp_move': '-1',
                'prev_outcome': 'none',
                'prev2_opp_move': '-1',
                'score_me_before': 0,
                'score_opp_before': 0,
                'streak_draws': 0,
                'stake': st.session_state.match_stats['stake'],
                'opp_match_wins': st.session_state.match_stats['opp_match_wins'],
                'opp_match_winrate': st.session_state.match_stats['opp_match_winrate'],
            }
            pred, your = predict_next_move(pipeline, le_target, features)
            st.info(f"🤖 Предсказание на первый раунд: противник – **{pred}**, вам – **{your}**")

        if submitted:
            # Вычисляем my_move по opp_move и outcome
            beat_map = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
            lose_map = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
            if outcome == 'win':
                my_move = beat_map[opp_move]
            elif outcome == 'lose':
                my_move = lose_map[opp_move]
            else:  # draw
                my_move = opp_move

            # Вычисляем счёт и streak_draws до этого раунда
            if not st.session_state.history:
                score_me_before = 0
                score_opp_before = 0
                streak_draws = 0
                prev_opp_move = '-1'
                prev_outcome = 'none'
                prev2_opp_move = '-1'
            else:
                last = st.session_state.history[-1]
                score_me_before = last['score_me_before'] + (1 if last['outcome'] == 'win' else 0)
                score_opp_before = last['score_opp_before'] + (1 if last['outcome'] == 'lose' else 0)
                # streak_draws для текущего раунда
                if last['outcome'] == 'draw':
                    streak_draws = last['streak_draws'] + 1
                else:
                    streak_draws = 0
                prev_opp_move = last['opp_move']
                prev_outcome = last['outcome']
                # prev2
                if len(st.session_state.history) >= 2:
                    prev2_opp_move = st.session_state.history[-2]['opp_move']
                else:
                    prev2_opp_move = '-1'

            # Создаём запись текущего раунда
            new_round = {
                'match_id': st.session_state.match_id,
                'round': st.session_state.round_num,
                'opp_match_wins': st.session_state.match_stats['opp_match_wins'],
                'opp_match_winrate': st.session_state.match_stats['opp_match_winrate'],
                'stake': st.session_state.match_stats['stake'],
                'opp_move': opp_move,
                'my_move': my_move,
                'outcome': outcome,
                'score_me_before': score_me_before,
                'score_opp_before': score_opp_before,
                'prev_opp_move': prev_opp_move,
                'prev_outcome': prev_outcome,
                'streak_draws': streak_draws,
                'prev2_opp_move': prev2_opp_move,
            }
            st.session_state.history.append(new_round)

            # Сохраняем в CSV
            df_new = pd.DataFrame([new_round])
            if os.path.exists(DATA_PATH):
                df_existing = pd.read_csv(DATA_PATH)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            df_combined.to_csv(DATA_PATH, index=False)

            # Обновляем счёт после раунда
            new_score_me = score_me_before + (1 if outcome == 'win' else 0)
            new_score_opp = score_opp_before + (1 if outcome == 'lose' else 0)

            # Проверяем окончание матча
            if new_score_me >= 3 or new_score_opp >= 3:
                st.success(f"Матч окончен! Итоговый счёт {new_score_me}:{new_score_opp}")
                if st.button("Начать новый матч"):
                    st.session_state.match_active = False
                    st.session_state.history = []
                    st.session_state.match_stats = {}
                    st.session_state.round_num = 1
                    st.rerun()
                st.stop()

            # Предсказываем следующий ход
            features_for_pred = {
                'opp_move': opp_move,
                'my_move': my_move,
                'outcome': outcome,
                'prev_opp_move': prev_opp_move,
                'prev_outcome': prev_outcome,
                'prev2_opp_move': prev2_opp_move,
                'score_me_before': score_me_before,
                'score_opp_before': score_opp_before,
                'streak_draws': streak_draws,
                'stake': st.session_state.match_stats['stake'],
                'opp_match_wins': st.session_state.match_stats['opp_match_wins'],
                'opp_match_winrate': st.session_state.match_stats['opp_match_winrate'],
            }
            pred, your = predict_next_move(pipeline, le_target, features_for_pred)
            st.success(f"🤖 Предсказание на следующий раунд: противник – **{pred}**, вам – **{your}**")

            # Увеличиваем номер раунда
            st.session_state.round_num += 1
            st.rerun()

    else:
        st.success("Матч завершён!")
        if st.button("Начать новый матч"):
            st.session_state.match_active = False
            st.session_state.history = []
            st.session_state.match_stats = {}
            st.session_state.round_num = 1
            st.rerun()

# ========================
# Кнопка для переобучения модели (опционально)
# ========================
st.markdown("---")
if st.button("🔄 Переобучить модель на всех данных"):
    st.warning("Функция переобучения требует отдельного скрипта. Запустите обучение локально.")