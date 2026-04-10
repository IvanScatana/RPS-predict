import streamlit as st
import pandas as pd
import joblib
import os

# ========================
# Конфигурация
# ========================
MODEL_PATH = 'rps_model.pkl'
TARGET_ENCODER_PATH = 'target_encoder.pkl'
DATA_PATH = 'rps_data.csv'

MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}

# ========================
# Очистка незавершённых матчей
# ========================
def clean_unfinished_matches():
    if not os.path.exists(DATA_PATH):
        return
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return
    finished_matches = set()
    for mid in df['match_id'].unique():
        match_df = df[df['match_id'] == mid]
        for _, row in match_df.iterrows():
            score_me_before = row['score_me_before']
            score_opp_before = row['score_opp_before']
            outcome = row['outcome']
            if (outcome == 'win' and score_me_before + 1 >= 3) or (outcome == 'lose' and score_opp_before + 1 >= 3):
                finished_matches.add(mid)
                break
    df_clean = df[df['match_id'].isin(finished_matches)]
    if len(df_clean) != len(df):
        df_clean.to_csv(DATA_PATH, index=False)
        st.cache_data.clear()

def get_next_match_id():
    if not os.path.exists(DATA_PATH):
        return 1
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return 1
    return df['match_id'].max() + 1

# ========================
# Загрузка модели
# ========================
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load(MODEL_PATH)
        le_target = joblib.load(TARGET_ENCODER_PATH)
        return pipeline, le_target
    except:
        st.error("Модель не найдена.")
        return None, None

def predict_next_move(pipeline, le_target, features_dict):
    input_df = pd.DataFrame([features_dict])
    pred_enc = pipeline.predict(input_df)[0]
    pred_move_letter = le_target.inverse_transform([pred_enc])[0]
    pred_move_full = LETTER_TO_MOVE[pred_move_letter]
    beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
    your_move_letter = beat[pred_move_letter]
    your_move_full = LETTER_TO_MOVE[your_move_letter]
    return pred_move_full, your_move_full

# ========================
# Инициализация сессии
# ========================
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'setup'
    st.session_state.match_id = None
    st.session_state.round_num = 1
    st.session_state.history = []
    st.session_state.score_me = 0
    st.session_state.score_opp = 0
    st.session_state.streak_draws = 0
    st.session_state.opp_stats = {'wins': 0, 'winrate': 0.5, 'stake': 25}

st.set_page_config(page_title="RPS Predictor", layout="centered")
st.title("🎮 Предсказатель хода в 'Камень-Ножницы-Бумага'")

clean_unfinished_matches()
pipeline, le_target = load_model()
if pipeline is None:
    st.stop()

# ========================
# 1. Настройка матча
# ========================
if st.session_state.game_state == 'setup':
    st.subheader("Новый матч")
    with st.form("setup_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            wins = st.number_input("Побед противника (матчи)", min_value=-1, step=1, value=0)
        with col2:
            winrate = st.number_input("Винрейт противника", min_value=-1.0, max_value=1.0, step=0.05, value=0.5)
        with col3:
            stake = st.selectbox("Ставка", [25, 50, 100], index=0)
        start = st.form_submit_button("Начать матч")
        if start:
            st.session_state.opp_stats = {'wins': wins, 'winrate': winrate, 'stake': stake}
            st.session_state.match_id = get_next_match_id()
            st.session_state.game_state = 'playing'
            st.session_state.round_num = 1
            st.session_state.history = []
            st.session_state.score_me = 0
            st.session_state.score_opp = 0
            st.session_state.streak_draws = 0
            st.rerun()

# ========================
# 2. Игровой процесс
# ========================
elif st.session_state.game_state == 'playing':
    st.info(f"📊 Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id}")

    # --- Предсказание для текущего раунда ---
    if len(st.session_state.history) == 0:
        # Первый раунд: нет предыдущей информации
        features = {
            'opp_move': 'К', 'my_move': 'К', 'outcome': 'draw',
            'prev_opp_move': '-1', 'prev_outcome': 'none', 'prev2_opp_move': '-1',
            'score_me_before': 0, 'score_opp_before': 0, 'streak_draws': 0,
            'stake': st.session_state.opp_stats['stake'],
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate'],
        }
        pred_move, your_move = predict_next_move(pipeline, le_target, features)
        st.success(f"🤖 Предсказание на **этот раунд**: противник – **{pred_move}**, вам – **{your_move}**")
    else:
        # Используем последний сыгранный раунд для предсказания следующего
        last = st.session_state.history[-1]
        features = {
            'opp_move': last['opp_move'],
            'my_move': last['my_move'],
            'outcome': last['outcome'],
            'prev_opp_move': last['prev_opp_move'],
            'prev_outcome': last['prev_outcome'],
            'prev2_opp_move': last['prev2_opp_move'],
            'score_me_before': st.session_state.score_me,
            'score_opp_before': st.session_state.score_opp,
            'streak_draws': st.session_state.streak_draws,
            'stake': st.session_state.opp_stats['stake'],
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate'],
        }
        pred_move, your_move = predict_next_move(pipeline, le_target, features)
        st.success(f"🤖 Предсказание на **этот раунд**: противник – **{pred_move}**, вам – **{your_move}**")

    # Форма ввода текущего раунда
    with st.form("round_form"):
        st.subheader(f"Введите данные раунда {st.session_state.round_num}")
        col1, col2 = st.columns(2)
        with col1:
            opp_move_full = st.selectbox("Ход противника", ["Камень", "Ножницы", "Бумага"], key="opp")
        with col2:
            outcome_ru = st.selectbox("Исход для вас", ["Победа", "Поражение", "Ничья"], key="out")
        submitted = st.form_submit_button("✅ Записать раунд")

    if submitted:
        opp_move = MOVE_TO_LETTER[opp_move_full]
        outcome = OUTCOME_TO_EN[outcome_ru]

        # Вычисляем свой ход
        beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
        lose = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
        if outcome == 'win':
            my_move = beat[opp_move]
        elif outcome == 'lose':
            my_move = lose[opp_move]
        else:
            my_move = opp_move

        prev_opp_move = st.session_state.history[-1]['opp_move'] if st.session_state.history else '-1'
        prev_outcome = st.session_state.history[-1]['outcome'] if st.session_state.history else 'none'
        prev2_opp_move = st.session_state.history[-2]['opp_move'] if len(st.session_state.history) >= 2 else '-1'

        new_round = {
            'match_id': st.session_state.match_id,
            'round': st.session_state.round_num,
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate'],
            'stake': st.session_state.opp_stats['stake'],
            'opp_move': opp_move,
            'my_move': my_move,
            'outcome': outcome,
            'score_me_before': st.session_state.score_me,
            'score_opp_before': st.session_state.score_opp,
            'prev_opp_move': prev_opp_move,
            'prev_outcome': prev_outcome,
            'streak_draws': st.session_state.streak_draws,
            'prev2_opp_move': prev2_opp_move,
        }
        st.session_state.history.append(new_round)

        # Обновляем счёт
        if outcome == 'win':
            st.session_state.score_me += 1
            st.session_state.streak_draws = 0
        elif outcome == 'lose':
            st.session_state.score_opp += 1
            st.session_state.streak_draws = 0
        else:
            st.session_state.streak_draws += 1

        # Сохраняем в CSV
        df_new = pd.DataFrame([new_round])
        if os.path.exists(DATA_PATH):
            df_old = pd.read_csv(DATA_PATH)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(DATA_PATH, index=False)
        st.cache_data.clear()

        # Проверяем окончание матча
        if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
            st.session_state.game_state = 'finished'
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        # Увеличиваем номер раунда и перезапускаем для следующего
        st.session_state.round_num += 1
        st.rerun()

# ========================
# 3. Завершение матча
# ========================
elif st.session_state.game_state == 'finished':
    st.info(f"📊 Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp}")
    if st.button("➕ Начать новый матч"):
        clean_unfinished_matches()
        st.session_state.game_state = 'setup'
        st.session_state.history = []
        st.session_state.score_me = 0
        st.session_state.score_opp = 0
        st.session_state.round_num = 1
        st.session_state.streak_draws = 0
        st.rerun()

# ========================
# История (все строки, не только последние)
# ========================
with st.expander("📜 История сохранённых раундов (завершённые матчи)"):
    if os.path.exists(DATA_PATH):
        df_view = pd.read_csv(DATA_PATH)
        if not df_view.empty:
            df_display = df_view.copy()
            for col in ['opp_move', 'my_move', 'prev_opp_move', 'prev2_opp_move']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
            if 'outcome' in df_display.columns:
                df_display['outcome'] = df_display['outcome'].map(EN_TO_OUTCOME)
            if 'prev_outcome' in df_display.columns:
                df_display['prev_outcome'] = df_display['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
            st.dataframe(df_display)  # показываем все строки
        else:
            st.write("Нет сохранённых данных.")
    else:
        st.write("Нет сохранённых данных.")