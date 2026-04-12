import streamlit as st
import pandas as pd
import os
import time
import json
from collections import defaultdict, Counter

# ========================
# Конфигурация
# ========================
STATS_FILE = 'rps_markov_stats.json'
DATA_PATH = 'rps_data.csv'

MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}
MOVE_EMOJI = {"Камень": "✊", "Ножницы": "✌️", "Бумага": "✋"}

# Ожидаемые колонки (теперь 17)
EXPECTED_COLS = [
    'match_id', 'round', 'player_name', 'win_category',
    'opp_match_wins', 'opp_match_winrate', 'stake',
    'opp_move', 'my_move', 'outcome',
    'score_me_before', 'score_opp_before', 'streak_draws',
    'prev_opp_move', 'prev_outcome', 'prev2_opp_move', 'prev2_outcome'
]

# ========================
# Функция для вычисления категории побед
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
# Марковский предиктор
# ========================
class MarkovRPSPredictor:
    def __init__(self):
        self.first_move_probs = {'К': 0.33, 'Н': 0.33, 'Б': 0.34}
        self.transitions = defaultdict(Counter)
        self.prev_opp_move = None

    def update(self, opp_move):
        if self.prev_opp_move is not None:
            self.transitions[self.prev_opp_move][opp_move] += 1
        self.prev_opp_move = opp_move

    def predict_proba(self):
        if self.prev_opp_move is None:
            total = sum(self.first_move_probs.values())
            return {k: v/total for k, v in self.first_move_probs.items()}
        counter = self.transitions[self.prev_opp_move]
        total = sum(counter.values())
        if total == 0:
            return {'К': 1/3, 'Н': 1/3, 'Б': 1/3}
        return {move: counter[move]/total for move in ['К', 'Н', 'Б']}

    def choose_opp_move(self):
        probs = self.predict_proba()
        return max(probs, key=probs.get)

    def choose_my_move(self):
        probs = self.predict_proba()
        pK, pH, pB = probs['К'], probs['Н'], probs['Б']
        ev = {'К': pH - pB, 'Н': pB - pK, 'Б': pK - pH}
        return max(ev, key=ev.get)

    def reset_match(self):
        self.prev_opp_move = None

    def save(self, path):
        data = {
            'first_move_probs': self.first_move_probs,
            'transitions': {k: dict(v) for k, v in self.transitions.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.first_move_probs = data.get('first_move_probs', self.first_move_probs)
                self.transitions = defaultdict(Counter)
                for k, v in data.get('transitions', {}).items():
                    self.transitions[k] = Counter(v)

# ========================
# Работа с CSV
# ========================
def load_data():
    """Загружает CSV с приведением типов, добавлением недостающих колонок и вычислением win_category."""
    numeric_cols = ['round', 'opp_match_wins', 'opp_match_winrate', 'stake',
                    'score_me_before', 'score_opp_before', 'streak_draws']
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
        # Приведение числовых колонок
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Приведение player_name к строке
        if 'player_name' in df.columns:
            df['player_name'] = df['player_name'].astype(str).replace('nan', '')
        else:
            df['player_name'] = ""
        # Добавление недостающих колонок (включая win_category)
        for col in EXPECTED_COLS:
            if col not in df.columns:
                if col == 'player_name':
                    df[col] = ""
                elif col == 'win_category':
                    # вычислим позже на основе opp_match_wins
                    continue
                elif col in numeric_cols:
                    df[col] = 0
                elif col in ['prev_outcome', 'prev2_outcome']:
                    df[col] = 'none'
                else:
                    df[col] = ""
        # Если win_category отсутствует, вычисляем её на основе opp_match_wins
        if 'win_category' not in df.columns:
            df['win_category'] = df['opp_match_wins'].apply(compute_win_category)
        else:
            # Пересчитываем на случай изменения правил (опционально)
            df['win_category'] = df['opp_match_wins'].apply(compute_win_category)
        # Приводим порядок колонок
        df = df[EXPECTED_COLS]
        return df
    except pd.errors.ParserError:
        st.warning("Файл данных повреждён, создаётся новый.")
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
    # Преобразование исходов
    if 'outcome' in df_last.columns:
        df_last['outcome'] = df_last['outcome'].map(EN_TO_OUTCOME)
    if 'prev_outcome' in df_last.columns:
        df_last['prev_outcome'] = df_last['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
    if 'prev2_outcome' in df_last.columns:
        df_last['prev2_outcome'] = df_last['prev2_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x) if x != 'none' else 'нет')
    return df_last

# ========================
# Инициализация сессии и предиктора
# ========================
def init_predictor():
    if 'predictor' not in st.session_state:
        st.session_state.predictor = MarkovRPSPredictor()
        st.session_state.predictor.load(STATS_FILE)
    return st.session_state.predictor

def predict_move(predictor, _=None):
    opp_letter = predictor.choose_opp_move()
    my_letter = predictor.choose_my_move()
    return LETTER_TO_MOVE[opp_letter], LETTER_TO_MOVE[my_letter]

# ========================
# Основное приложение
# ========================
st.set_page_config(page_title="Помощник в игре Камень - Ножницы - Бумага", layout="wide")
st.title("🎮 Помощник в игре 'Камень - Ножницы - Бумага'")

ensure_csv()
predictor = init_predictor()

# Состояния сессии
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
            predictor.reset_match()
            pred_move, your_move = predict_move(predictor)
            st.session_state.next_prediction = (pred_move, your_move)
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

        # Вычисляем win_category на основе opp_match_wins
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
            'prev_opp_move': prev_opp,
            'prev_outcome': prev_out,
            'streak_draws': st.session_state.streak_draws,
            'prev2_opp_move': prev2_opp,
            'prev2_outcome': prev2_out
        }
        st.session_state.history.append(new_row)

        predictor.update(opp_letter)

        if outcome == 'win':
            st.session_state.score_me += 1
            st.session_state.streak_draws = 0
        elif outcome == 'lose':
            st.session_state.score_opp += 1
            st.session_state.streak_draws = 0
        else:
            st.session_state.streak_draws += 1

        df_new = pd.DataFrame([new_row])
        if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
            df_new.to_csv(DATA_PATH, index=False, sep=',', encoding='utf-8')
        else:
            df_new.to_csv(DATA_PATH, mode='a', header=False, index=False, sep=',', encoding='utf-8')
        st.cache_data.clear()
        time.sleep(0.1)

        if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
            st.session_state.game_state = 'finished'
            st.session_state.next_prediction = None
            predictor.save(STATS_FILE)
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        pred_move, your_move = predict_move(predictor)
        st.session_state.next_prediction = (pred_move, your_move)

        st.session_state.selected_opp = None
        st.session_state.selected_outcome = None
        st.session_state.round_num += 1
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