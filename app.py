import streamlit as st
import pandas as pd
import joblib
import os

# ---------- Конфигурация ----------
MODEL_PATH = 'rps_model.pkl'          # обученная модель
TARGET_ENCODER_PATH = 'target_encoder.pkl'
DATA_PATH = 'rps_data.csv'

# Словари перевода
MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}

# ---------- Функции работы с CSV ----------
def init_csv():
    """Создаёт CSV с заголовками, если файла нет."""
    if not os.path.exists(DATA_PATH):
        columns = ['match_id', 'round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                   'opp_move', 'my_move', 'outcome', 'score_me_before', 'score_opp_before',
                   'prev_opp_move', 'prev_outcome', 'streak_draws', 'prev2_opp_move']
        pd.DataFrame(columns=columns).to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')

def append_round(round_data):
    """Добавляет одну строку в CSV (быстрое добавление в конец)."""
    df_new = pd.DataFrame([round_data])
    # Если файл пуст или нет заголовков, запишем с заголовками
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        df_new.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
    else:
        # Добавляем в конец без заголовка
        df_new.to_csv(DATA_PATH, mode='a', header=False, index=False, sep=',', encoding='utf-8')

def get_last_match_id():
    """Возвращает максимальный match_id из существующих записей, или 0."""
    if not os.path.exists(DATA_PATH):
        return 0
    df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
    if df.empty:
        return 0
    return df['match_id'].max()

# ---------- Загрузка модели ----------
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load(MODEL_PATH)
        le_target = joblib.load(TARGET_ENCODER_PATH)
        return pipeline, le_target
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

def predict_next(pipeline, le_target, features_dict):
    """Возвращает (предсказанный ход, ваш оптимальный ход) в виде полных названий."""
    input_df = pd.DataFrame([features_dict])
    pred_enc = pipeline.predict(input_df)[0]
    pred_letter = le_target.inverse_transform([pred_enc])[0]
    pred_move = LETTER_TO_MOVE[pred_letter]
    beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
    your_letter = beat[pred_letter]
    your_move = LETTER_TO_MOVE[your_letter]
    return pred_move, your_move

# ---------- Инициализация состояния ----------
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'setup'   # setup, playing, finished
    st.session_state.match_id = None
    st.session_state.round_num = 1
    st.session_state.history = []           # список словарей раундов текущего матча
    st.session_state.score_me = 0
    st.session_state.score_opp = 0
    st.session_state.streak_draws = 0
    st.session_state.opp_stats = {'wins': 0, 'winrate': 0.5, 'stake': 25}
    st.session_state.next_prediction = None   # (pred_move, your_move) для текущего раунда

st.set_page_config(page_title="RPS Predictor", layout="centered")
st.title("🎮 Предсказатель хода в 'Камень-Ножницы-Бумага'")

init_csv()
pipeline, le_target = load_model()
if pipeline is None:
    st.stop()

# ---------- 1. Настройка нового матча ----------
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
            # Определяем новый match_id
            last_id = get_last_match_id()
            st.session_state.match_id = last_id + 1
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
            pred, your = predict_next(pipeline, le_target, feats)
            st.session_state.next_prediction = (pred, your)
            st.rerun()

# ---------- 2. Игровой процесс ----------
elif st.session_state.game_state == 'playing':
    st.info(f"Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id}")

    # Отображаем предсказание для текущего раунда
    if st.session_state.next_prediction:
        pred_move, your_move = st.session_state.next_prediction
        st.success(f"🤖 Предсказание на **раунд {st.session_state.round_num}**: противник – **{pred_move}**, вам – **{your_move}**")
    else:
        st.info("Предсказание недоступно")

    # Форма ввода данных сыгранного раунда
    with st.form("round_form"):
        st.subheader(f"Введите данные раунда {st.session_state.round_num}")
        col1, col2 = st.columns(2)
        with col1:
            opp_move_full = st.selectbox("Ход противника", ["Камень", "Ножницы", "Бумага"], key="opp")
        with col2:
            outcome_ru = st.selectbox("Исход для вас", ["Победа", "Поражение", "Ничья"], key="out")
        submitted = st.form_submit_button("✅ Записать раунд")

    if submitted:
        opp_letter = MOVE_TO_LETTER[opp_move_full]
        outcome = OUTCOME_TO_EN[outcome_ru]

        # Вычисляем свой ход
        beat = {'К': 'Б', 'Н': 'К', 'Б': 'Н'}
        lose = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
        if outcome == 'win':
            my_letter = beat[opp_letter]
        elif outcome == 'lose':
            my_letter = lose[opp_letter]
        else:
            my_letter = opp_letter

        # Предыдущие данные из истории текущего матча
        if st.session_state.history:
            prev_opp = st.session_state.history[-1]['opp_move']
            prev_out = st.session_state.history[-1]['outcome']
            prev2_opp = st.session_state.history[-2]['opp_move'] if len(st.session_state.history) >= 2 else '-1'
        else:
            prev_opp = '-1'
            prev_out = 'none'
            prev2_opp = '-1'

        # Формируем запись раунда
        round_record = {
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
        st.session_state.history.append(round_record)
        # Сохраняем в CSV
        append_round(round_record)

        # Обновляем счёт и серию ничьих
        if outcome == 'win':
            st.session_state.score_me += 1
            st.session_state.streak_draws = 0
        elif outcome == 'lose':
            st.session_state.score_opp += 1
            st.session_state.streak_draws = 0
        else:
            st.session_state.streak_draws += 1

        # Проверка окончания матча
        if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
            st.session_state.game_state = 'finished'
            st.session_state.next_prediction = None
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        # Предсказание для следующего раунда
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
        pred, your = predict_next(pipeline, le_target, feats)
        st.session_state.next_prediction = (pred, your)

        st.session_state.round_num += 1
        st.rerun()

# ---------- 3. Завершение матча ----------
elif st.session_state.game_state == 'finished':
    st.info(f"Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp}")
    if st.button("➕ Начать новый матч"):
        st.session_state.game_state = 'setup'
        st.session_state.history = []
        st.session_state.score_me = 0
        st.session_state.score_opp = 0
        st.session_state.round_num = 1
        st.session_state.streak_draws = 0
        st.session_state.next_prediction = None
        st.rerun()

# ---------- 4. Просмотр истории (все раунды) ----------
with st.expander("📜 История всех сохранённых раундов"):
    if os.path.exists(DATA_PATH):
        df_view = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
        if not df_view.empty:
            # Преобразуем для отображения
            df_disp = df_view.copy()
            for col in ['opp_move', 'my_move', 'prev_opp_move', 'prev2_opp_move']:
                if col in df_disp.columns:
                    df_disp[col] = df_disp[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
            if 'outcome' in df_disp.columns:
                df_disp['outcome'] = df_disp['outcome'].map(EN_TO_OUTCOME)
            if 'prev_outcome' in df_disp.columns:
                df_disp['prev_outcome'] = df_disp['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
            st.dataframe(df_disp.tail(30))
        else:
            st.write("Файл истории пуст.")
    else:
        st.write("Файл истории не найден.")

with st.expander("💾 Скачать историю"):
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            csv_data = f.read()
        st.download_button(
            label="📥 Скачать rps_data.csv",
            data=csv_data,
            file_name="rps_data.csv",
            mime="text/csv"
        )
    else:
        st.write("Файл ещё не создан.")