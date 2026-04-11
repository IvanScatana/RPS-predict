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

# Эмодзи для ходов
MOVE_EMOJI = {"Камень": "✊", "Ножницы": "✌️", "Бумага": "✋"}

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

def get_last_n_records(n=10):
    """Возвращает последние n записей из CSV в читаемом русском виде."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
    if df.empty:
        return df
    df_last = df.tail(n).copy()
    # Перевод
    for col in ['opp_move', 'my_move', 'prev_opp_move', 'prev2_opp_move']:
        if col in df_last.columns:
            df_last[col] = df_last[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
    if 'outcome' in df_last.columns:
        df_last['outcome'] = df_last['outcome'].map(EN_TO_OUTCOME)
    if 'prev_outcome' in df_last.columns:
        df_last['prev_outcome'] = df_last['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
    return df_last

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
    st.session_state.next_prediction = None
    st.session_state.selected_opp = None
    st.session_state.selected_outcome = None

st.set_page_config(page_title="Помощник в игре Камень - Ножницы - Бумага", layout="wide")
st.title("🎮 Помощник в игре 'Камень - Ножницы - Бумага'")

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
            winrate_percent = st.number_input("Винрейт противника (%)", min_value=-100.0, max_value=100.0, step=0.01, value=50.0)
            winrate = winrate_percent / 100.0
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
            st.session_state.selected_opp = None
            st.session_state.selected_outcome = None
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
    # Основной игровой интерфейс (две колонки)
    col_game, col_history = st.columns([2, 1])

    with col_game:
        st.info(f"Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id}")

        if st.session_state.next_prediction:
            pred_move, your_move = st.session_state.next_prediction
            st.success(f"🤖 Предсказание на **раунд {st.session_state.round_num}**: противник – **{pred_move}**, вам – **{your_move}**")

        st.subheader("Выберите ход противника и исход раунда")

        # Кнопки для хода противника
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

        # Кнопки для исхода
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

        # Кнопка "Следующий раунд"
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

            if outcome == 'win':
                st.session_state.score_me += 1
                st.session_state.streak_draws = 0
            elif outcome == 'lose':
                st.session_state.score_opp += 1
                st.session_state.streak_draws = 0
            else:
                st.session_state.streak_draws += 1

            df_new = pd.DataFrame([new_row])
            if os.path.exists(DATA_PATH):
                existing = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
                df_combined = pd.concat([existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            df_combined.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
            st.cache_data.clear()

            if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
                st.session_state.game_state = 'finished'
                st.session_state.next_prediction = None
                st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
                st.rerun()

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

            st.session_state.selected_opp = None
            st.session_state.selected_outcome = None
            st.session_state.round_num += 1
            st.rerun()

        # ========== ВИЗУАЛЬНОЕ ОТОБРАЖЕНИЕ ХОДОВ ПРОТИВНИКА В ТЕКУЩЕЙ ИГРЕ ==========
        st.markdown("---")
        st.subheader("📊 Ходы противника в текущем матче")
        if st.session_state.history:
            # Преобразуем буквы в названия и эмодзи
            moves = []
            for rec in st.session_state.history:
                move_letter = rec['opp_move']
                move_name = LETTER_TO_MOVE.get(move_letter, "?")
                emoji = MOVE_EMOJI.get(move_name, "❓")
                moves.append(f"{emoji} {move_name}")
            st.write(" → ".join(moves))
        else:
            st.write("Пока нет записанных ходов.")

    with col_history:
        # ========== ПОСЛЕДНИЕ 10 ЗАПИСЕЙ ИЗ CSV ==========
        st.subheader("📋 Последние 10 сохранённых раундов")
        last_records = get_last_n_records(10)
        if not last_records.empty:
            # Показываем только нужные колонки для краткости
            show_cols = ['match_id', 'round', 'opp_move', 'my_move', 'outcome', 'score_me_before', 'score_opp_before']
            available = [c for c in show_cols if c in last_records.columns]
            st.dataframe(last_records[available], use_container_width=True, height=400)
        else:
            st.write("Нет записей. После первого раунда данные появятся.")

# ========================
# Завершение матча
# ========================
elif st.session_state.game_state == 'finished':
    st.info(f"Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp}")
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

# ========================
# Дополнительная кнопка скачивания CSV (опционально)
# ========================
with st.expander("💾 Скачать историю (CSV)"):
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            csv_data = f.read()
        st.download_button("📥 Скачать rps_data.csv", data=csv_data, file_name="rps_data.csv", mime="text/csv")
    else:
        st.write("Файл ещё не создан.")