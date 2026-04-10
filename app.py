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
# Работа с CSV
# ========================
def ensure_csv():
    expected_cols = ['match_id', 'round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                     'opp_move', 'my_move', 'outcome', 'score_me_before', 'score_opp_before',
                     'prev_opp_move', 'prev_outcome', 'streak_draws', 'prev2_opp_move']
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=expected_cols).to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')

def clean_unfinished():
    if not os.path.exists(DATA_PATH):
        return
    df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
    if df.empty:
        return
    finished = set()
    for mid in df['match_id'].unique():
        match = df[df['match_id'] == mid]
        for _, row in match.iterrows():
            if (row['outcome'] == 'win' and row['score_me_before'] + 1 >= 3) or \
               (row['outcome'] == 'lose' and row['score_opp_before'] + 1 >= 3):
                finished.add(mid)
                break
    df_clean = df[df['match_id'].isin(finished)]
    if len(df_clean) != len(df):
        df_clean.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
        st.cache_data.clear()

def next_match_id():
    if not os.path.exists(DATA_PATH):
        return 1
    df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
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
        st.error("Модель не найдена. Обучите и сохраните rps_model.pkl и target_encoder.pkl")
        return None, None

def predict_move(pipeline, le_target, features):
    inp = pd.DataFrame([features])
    pred = pipeline.predict(inp)[0]
    letter = le_target.inverse_transform([pred])[0]
    move = LETTER_TO_MOVE[letter]
    beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
    your = LETTER_TO_MOVE[beat[letter]]
    return move, your

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
    st.session_state.next_prediction = None  # (pred_move, your_move)

st.set_page_config(page_title="RPS Predictor", layout="centered")
st.title("🎮 Предсказатель хода в 'Камень-Ножницы-Бумага'")

ensure_csv()
clean_unfinished()
pipeline, le_target = load_model()
if pipeline is None:
    st.stop()

# ========================
# Начало матча
# ========================
if st.session_state.game_state == 'setup':
    st.subheader("Новый матч")
    with st.form("setup"):
        c1, c2, c3 = st.columns(3)
        with c1:
            wins = st.number_input("Побед противника (матчи)", min_value=-1, step=1, value=0)
        with c2:
            winrate = st.number_input("Винрейт противника", min_value=-1.0, max_value=1.0, step=0.05, value=0.5)
        with c3:
            stake = st.selectbox("Ставка", [25, 50, 100])
        if st.form_submit_button("Начать матч"):
            st.session_state.opp_stats = {'wins': wins, 'winrate': winrate, 'stake': stake}
            st.session_state.match_id = next_match_id()
            st.session_state.game_state = 'playing'
            st.session_state.round_num = 1
            st.session_state.history = []
            st.session_state.score_me = 0
            st.session_state.score_opp = 0
            st.session_state.streak_draws = 0
            # Предсказание для первого раунда
            feats = {
                'opp_move': 'К', 'my_move': 'К', 'outcome': 'draw',
                'prev_opp_move': '-1', 'prev_outcome': 'none', 'prev2_opp_move': '-1',
                'score_me_before': 0, 'score_opp_before': 0, 'streak_draws': 0,
                'stake': stake, 'opp_match_wins': wins, 'opp_match_winrate': winrate
            }
            pred, your = predict_move(pipeline, le_target, feats)
            st.session_state.next_prediction = (pred, your)
            st.rerun()

# ========================
# Игровой процесс
# ========================
elif st.session_state.game_state == 'playing':
    st.info(f"Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id}")

    # Показываем предсказание для текущего раунда (если есть)
    if st.session_state.next_prediction:
        pred_move, your_move = st.session_state.next_prediction
        st.success(f"🤖 Предсказание на **раунд {st.session_state.round_num}**: противник – **{pred_move}**, вам – **{your_move}**")
    else:
        st.info("Предсказание загружается...")

    with st.form("round"):
        st.subheader(f"Введите данные раунда {st.session_state.round_num}")
        col1, col2 = st.columns(2)
        with col1:
            opp = st.selectbox("Ход противника", ["Камень", "Ножницы", "Бумага"], key="opp")
        with col2:
            out = st.selectbox("Исход для вас", ["Победа", "Поражение", "Ничья"], key="out")
        submitted = st.form_submit_button("✅ Записать раунд")

    if submitted:
        opp_letter = MOVE_TO_LETTER[opp]
        outcome = OUTCOME_TO_EN[out]

        # Вычисляем свой ход
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

        new_row = {
            'match_id': st.session_state.match_id,
            'round': st.session_state.round_num,
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate'],
            'stake': st.session_state.opp_stats['stake'],
            'opp_move': opp_letter,
            'my_move': my_letter,
            'outcome': outcome,
            'score_me_before': st.session_state.score_me,
            'score_opp_before': st.session_state.score_opp,
            'prev_opp_move': prev_opp,
            'prev_outcome': prev_out,
            'streak_draws': st.session_state.streak_draws,
            'prev2_opp_move': prev2_opp
        }
        st.session_state.history.append(new_row)

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
        df_new = pd.DataFrame([new_row])
        if os.path.exists(DATA_PATH):
            existing = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
            df_combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
        st.cache_data.clear()

        # Проверка окончания матча
        if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
            st.session_state.game_state = 'finished'
            st.session_state.next_prediction = None
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        # Вычисляем предсказание для следующего раунда
        feats = {
            'opp_move': opp_letter,
            'my_move': my_letter,
            'outcome': outcome,
            'prev_opp_move': prev_opp,
            'prev_outcome': prev_out,
            'prev2_opp_move': prev2_opp,
            'score_me_before': st.session_state.score_me,
            'score_opp_before': st.session_state.score_opp,
            'streak_draws': st.session_state.streak_draws,
            'stake': st.session_state.opp_stats['stake'],
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate']
        }
        pred, your = predict_move(pipeline, le_target, feats)
        st.session_state.next_prediction = (pred, your)

        st.session_state.round_num += 1
        st.rerun()

# ========================
# Завершение матча
# ========================
elif st.session_state.game_state == 'finished':
    st.info(f"Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp}")
    if st.button("➕ Начать новый матч"):
        clean_unfinished()
        st.session_state.game_state = 'setup'
        st.session_state.history = []
        st.session_state.score_me = 0
        st.session_state.score_opp = 0
        st.session_state.round_num = 1
        st.session_state.streak_draws = 0
        st.session_state.next_prediction = None
        st.rerun()

# ========================
# История
# ========================
with st.expander("📜 История сохранённых раундов (завершённые матчи)"):
    if os.path.exists(DATA_PATH):
        df_view = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
        if not df_view.empty:
            df_disp = df_view.copy()
            for col in ['opp_move', 'my_move', 'prev_opp_move', 'prev2_opp_move']:
                if col in df_disp.columns:
                    df_disp[col] = df_disp[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
            if 'outcome' in df_disp.columns:
                df_disp['outcome'] = df_disp['outcome'].map(EN_TO_OUTCOME)
            if 'prev_outcome' in df_disp.columns:
                df_disp['prev_outcome'] = df_disp['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
            st.dataframe(df_disp.tail(20))
        else:
            st.write("Нет сохранённых данных.")
    else:
        st.write("Файл истории не найден.")