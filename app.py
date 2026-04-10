import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        return None, None

# ========================
# Функция предсказания
# ========================
def predict_next_move(pipeline, le_target, input_dict):
    """
    input_dict: словарь с ключами:
        'opp_move', 'my_move', 'outcome', 'prev_opp_move', 'prev_outcome', 'prev2_opp_move',
        'opp_match_wins', 'opp_match_winrate', 'stake',
        'score_me_before', 'score_opp_before', 'streak_draws'
    """
    input_df = pd.DataFrame([input_dict])
    pred_enc = pipeline.predict(input_df)[0]
    pred_move = le_target.inverse_transform([pred_enc])[0]
    beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
    your_move = beat[pred_move]
    return pred_move, your_move

# ========================
# Интерфейс
# ========================
st.set_page_config(page_title="RPS Predictor", layout="centered")
st.title("🎮 Предсказатель хода в 'Камень-Ножницы-Бумага'")
st.markdown("Введите данные о текущем раунде, и модель предскажет следующий ход противника.")

# Загрузка модели
pipeline, le_target = load_model()
if pipeline is None:
    st.error("Модель не найдена. Сначала обучите модель и сохраните её как 'rps_model.pkl'.")
    st.stop()

# Форма ввода
with st.form("prediction_form"):
    st.subheader("📊 Текущая ситуация")
    col1, col2, col3 = st.columns(3)
    with col1:
        prev2_opp_move = st.selectbox("Ход противника 2 раунда назад", ["-1", "К", "Н", "Б"], index=0)
        prev_opp_move = st.selectbox("Предыдущий ход противника", ["-1", "К", "Н", "Б"], index=0)
        prev_outcome = st.selectbox("Исход предыдущего раунда", ["none", "win", "lose", "draw"], index=0)
    with col2:
        opp_move = st.selectbox("Ход противника в текущем раунде", ["К", "Н", "Б"])
        my_move = st.selectbox("Ваш ход в текущем раунде", ["К", "Н", "Б"])
        outcome = st.selectbox("Исход текущего раунда", ["win", "lose", "draw"])
    with col3:
        score_me_before = st.number_input("Ваши победы (до раунда)", min_value=0, max_value=2, step=1)
        score_opp_before = st.number_input("Победы противника (до раунда)", min_value=0, max_value=2, step=1)
        streak_draws = st.number_input("Серия ничьих подряд", min_value=0, step=1)
        stake = st.selectbox("Ставка", [25, 50, 100], index=0)
        opp_match_wins = st.number_input("Побед противника в матчах (ист.)", value=0, step=1)
        opp_match_winrate = st.number_input("Винрейт противника (ист.)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    submitted = st.form_submit_button("🔮 Предсказать следующий ход")

    if submitted:
        input_dict = {
            'opp_move': opp_move,
            'my_move': my_move,
            'outcome': outcome,
            'prev_opp_move': prev_opp_move,
            'prev_outcome': prev_outcome,
            'prev2_opp_move': prev2_opp_move,
            'opp_match_wins': opp_match_wins,
            'opp_match_winrate': opp_match_winrate,
            'stake': stake,
            'score_me_before': score_me_before,
            'score_opp_before': score_opp_before,
            'streak_draws': streak_draws
        }
        pred_move, your_move = predict_next_move(pipeline, le_target, input_dict)
        st.success(f"🤖 Модель предсказывает следующий ход противника: **{pred_move}**")
        st.info(f"💡 Ваш оптимальный ответ: **{your_move}**")

# ========================
# Добавление новых данных (опционально)
# ========================
st.markdown("---")
st.header("➕ Добавить сыгранный раунд в базу данных")

with st.form("add_data_form"):
    st.subheader("Заполните данные завершённого раунда")
    col1, col2 = st.columns(2)
    with col1:
        match_id = st.number_input("ID матча", min_value=1, step=1)
        round_num = st.number_input("Номер раунда", min_value=1, step=1)
        opp_move_new = st.selectbox("Ход противника", ["К", "Н", "Б"], key="opp_new")
        my_move_new = st.selectbox("Ваш ход", ["К", "Н", "Б"], key="my_new")
    with col2:
        outcome_new = st.selectbox("Исход для вас", ["win", "lose", "draw"], key="out_new")
        score_me_before_new = st.number_input("Ваши победы до раунда", min_value=0, max_value=2, step=1, key="score_me_new")
        score_opp_before_new = st.number_input("Победы противника до раунда", min_value=0, max_value=2, step=1, key="score_opp_new")
        streak_draws_new = st.number_input("Серия ничьих подряд", min_value=0, step=1, key="streak_new")
    prev_opp_move_new = st.text_input("Предыдущий ход противника (или -1)", value="-1")
    prev_outcome_new = st.selectbox("Предыдущий исход", ["none", "win", "lose", "draw"], key="prev_out_new")
    prev2_opp_move_new = st.text_input("Ход противника 2 раунда назад (или -1)", value="-1")
    stake_new = st.selectbox("Ставка", [25, 50, 100], key="stake_new")
    opp_match_wins_new = st.number_input("Побед противника в матчах (ист.)", value=0, step=1, key="wins_new")
    opp_match_winrate_new = st.number_input("Винрейт противника (ист.)", min_value=0.0, max_value=1.0, step=0.05, value=0.5, key="wr_new")

    submitted_add = st.form_submit_button("💾 Сохранить раунд")

    if submitted_add:
        # Загружаем существующий CSV, добавляем строку, сохраняем
        try:
            df = pd.read_csv(DATA_PATH)
        except:
            df = pd.DataFrame(columns=['match_id', 'round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                                       'opp_move', 'my_move', 'outcome', 'score_me_before', 'score_opp_before',
                                       'prev_opp_move', 'prev_outcome', 'streak_draws', 'prev2_opp_move'])
        new_row = pd.DataFrame([{
            'match_id': match_id,
            'round': round_num,
            'opp_match_wins': opp_match_wins_new,
            'opp_match_winrate': opp_match_winrate_new,
            'stake': stake_new,
            'opp_move': opp_move_new,
            'my_move': my_move_new,
            'outcome': outcome_new,
            'score_me_before': score_me_before_new,
            'score_opp_before': score_opp_before_new,
            'prev_opp_move': prev_opp_move_new,
            'prev_outcome': prev_outcome_new,
            'streak_draws': streak_draws_new,
            'prev2_opp_move': prev2_opp_move_new
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success("✅ Раунд добавлен. Для обновления модели перезапустите обучение.")
        st.info("⚠️ Не забудьте переобучить модель, чтобы учитывать новые данные.")

# ========================
# Кнопка переобучения модели (запуск внешнего скрипта или встроенная функция)
# ========================
st.markdown("---")
if st.button("🔄 Переобучить модель на всех данных"):
    st.warning("Эта функция требует отдельного скрипта обучения. Реализуйте её по необходимости.")
    # Здесь можно вызвать subprocess или импортировать функцию обучения.