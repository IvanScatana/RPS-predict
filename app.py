import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import pickle
from catboost import CatBoostClassifier

# ========================
# Конфигурация
# ========================
DATA_PATH = 'rps_data.csv'
MODEL_PATH = 'cb_optimal.cbm'
GLOBAL_FREQ_PATH = 'global_freq.pkl'

MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}
MOVE_EMOJI = {"Камень": "✊", "Ножницы": "✌️", "Бумага": "✋"}

# Параметры CatBoost (проверенные)
CB_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 4,
    'loss_function': 'MultiClass',
    'random_seed': 42,
    'verbose': False
}

# Колонки (как в вашем CSV)
BASE_COLS = [
    'match_id', 'round', 'player_name', 'win_category',
    'opp_match_wins', 'opp_match_winrate', 'stake',
    'opp_move', 'my_move', 'outcome',
    'score_me_before', 'score_opp_before', 'score_diff', 'streak_draws'
]
PREV_COLS = []
for shift in range(1, 7):
    PREV_COLS.extend([f'prev{shift}_opp_move', f'prev{shift}_my_move', f'prev{shift}_outcome'])
EXPECTED_COLS = BASE_COLS + ['is_last_round'] + PREV_COLS

# ========================
# Вспомогательные функции (без изменений)
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
    if (my_move == 'К' and opp_move == 'Н') or \
       (my_move == 'Н' and opp_move == 'Б') or \
       (my_move == 'Б' and opp_move == 'К'):
        return 'win'
    return 'loss'

@st.cache_data(ttl=3600)
def load_data_cached():
    # ... (ваша исходная функция без изменений)
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
        df = df[[c for c in EXPECTED_COLS if c in df.columns]]
        return df
    except Exception as e:
        st.warning(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def ensure_csv():
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(DATA_PATH, index=False)
    else:
        df = load_data_cached()
        df.to_csv(DATA_PATH, index=False)

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
            if (outcome == 'win' and score_me + 1 >= 3) or \
               (outcome == 'lose' and score_opp + 1 >= 3):
                finished_matches.add(mid)
                break
    df_clean = df[df['match_id'].isin(finished_matches)]
    if len(df_clean) != len(df):
        df_clean.to_csv(DATA_PATH, index=False)
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

# ========================
# Генератор признаков для ML
# ========================
def create_features(df, global_opp_freq=None):
    df = df.copy()
    df['is_last_round'] = df['is_last_round'].astype(int)
    df['known_player'] = (df['player_name'] != 'Неизвестен').astype(int)
    win_cat_map = {'unknown': 0, 'до 5 побед': 1, '5-20': 2, '20-100': 3, '100+': 4}
    df['win_cat_code'] = df['win_category'].map(win_cat_map).fillna(0).astype(int)
    df['opp_wins_missing'] = (df['opp_match_wins'] == -1).astype(int)
    df['opp_winrate_missing'] = (df['opp_match_winrate'] == -0.01).astype(int)

    alpha = 0.7
    for n in [1,2,3,4,5,6]:
        weights = np.array([alpha**i for i in range(n)])
        weights = weights / weights.sum()
        for move in ['К','Н','Б']:
            cnt = np.zeros(len(df))
            for i in range(n):
                prev_col = f'prev{i+1}_opp_move'
                if prev_col in df.columns:
                    cnt += (df[prev_col] == move).astype(int) * weights[i]
            df[f'opp_exp_freq_{move}_last{n}'] = cnt

    for n in [1,2,3,4,5,6]:
        weights = np.array([alpha**i for i in range(n)])
        weights = weights / weights.sum()
        for move in ['К','Н','Б']:
            cnt = np.zeros(len(df))
            for i in range(n):
                prev_col = f'prev{i+1}_my_move'
                if prev_col in df.columns:
                    cnt += (df[prev_col] == move).astype(int) * weights[i]
            df[f'my_exp_freq_{move}_last{n}'] = cnt

    df['opp_bigram'] = 'none'
    mask = df['prev2_opp_move'].notna() & df['prev1_opp_move'].notna()
    df.loc[mask, 'opp_bigram'] = df.loc[mask, 'prev2_opp_move'].astype(str)+'_'+df.loc[mask, 'prev1_opp_move'].astype(str)

    df['my_bigram'] = 'none'
    mask_my = df['prev2_my_move'].notna() & df['prev1_my_move'].notna()
    df.loc[mask_my, 'my_bigram'] = df.loc[mask_my, 'prev2_my_move'].astype(str)+'_'+df.loc[mask_my, 'prev1_my_move'].astype(str)

    df['triple'] = 'none'
    mask_tr = df['prev1_opp_move'].notna() & df['prev1_my_move'].notna() & df['prev1_outcome'].notna()
    df.loc[mask_tr, 'triple'] = (df.loc[mask_tr, 'prev1_opp_move'].astype(str) + '_' +
                                 df.loc[mask_tr, 'prev1_my_move'].astype(str) + '_' +
                                 df.loc[mask_tr, 'prev1_outcome'].astype(str))

    df['same_opp_as_prev1'] = (df['prev1_opp_move'] == df['prev2_opp_move']).astype(int)
    df['same_my_as_prev1'] = (df['prev1_my_move'] == df['prev2_my_move']).astype(int)

    df['score_diff'] = df['score_me_before'] - df['score_opp_before']

    if global_opp_freq is not None:
        for col in ['global_opp_К','global_opp_Н','global_opp_Б']:
            if col in df.columns: df.drop(columns=col, inplace=True)
        df = df.join(global_opp_freq, on='player_name')
        df[['global_opp_К','global_opp_Н','global_opp_Б']] = \
            df[['global_opp_К','global_opp_Н','global_opp_Б']].fillna(0.0)

    return df

def prepare_features(df):
    exclude = ['match_id', 'opp_move', 'my_move', 'outcome', 'player_name', 'win_category',
               'prev1_my_move', 'prev2_my_move', 'prev3_my_move',
               'prev4_my_move', 'prev5_my_move', 'prev6_my_move',
               'score_me_before', 'score_opp_before']
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_feats = ['stake', 'prev1_opp_move', 'prev2_opp_move', 'prev3_opp_move',
                 'prev4_opp_move', 'prev5_opp_move', 'prev6_opp_move',
                 'opp_bigram', 'my_bigram', 'triple',
                 'prev1_outcome', 'prev2_outcome', 'prev3_outcome',
                 'prev4_outcome', 'prev5_outcome', 'prev6_outcome']
    for col in cat_feats:
        df[col] = df[col].astype('category')
    return df[feature_cols], cat_feats

# ========================
# Загрузка / обучение модели
# ========================
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(GLOBAL_FREQ_PATH):
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        with open(GLOBAL_FREQ_PATH, 'rb') as f:
            global_freq = pickle.load(f)
    else:
        df = load_data_cached()
        # Глобальные частоты
        global_freq = df.groupby('player_name')['opp_move'].value_counts(normalize=True).unstack(fill_value=0)
        for m in ['К','Н','Б']:
            if m not in global_freq.columns:
                global_freq[m] = 0.0
        global_freq = global_freq[['К','Н','Б']]
        global_freq.columns = ['global_opp_К','global_opp_Н','global_opp_Б']

        df_feat = create_features(df, global_freq)
        X, cat_feats = prepare_features(df_feat)
        y = df['opp_move'].map({'К': 1, 'Н': 2, 'Б': 0}).values  # optimal move idx

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(X, y, cat_features=cat_feats)
        model.save_model(MODEL_PATH)
        with open(GLOBAL_FREQ_PATH, 'wb') as f:
            pickle.dump(global_freq, f)

    with open(GLOBAL_FREQ_PATH, 'rb') as f:
        global_freq = pickle.load(f)
    return model, global_freq

def retrain_model():
    df = load_data_cached()
    global_freq = df.groupby('player_name')['opp_move'].value_counts(normalize=True).unstack(fill_value=0)
    for m in ['К','Н','Б']:
        if m not in global_freq.columns:
            global_freq[m] = 0.0
    global_freq = global_freq[['К','Н','Б']]
    global_freq.columns = ['global_opp_К','global_opp_Н','global_opp_Б']

    df_feat = create_features(df, global_freq)
    X, cat_feats = prepare_features(df_feat)
    y = df['opp_move'].map({'К': 1, 'Н': 2, 'Б': 0}).values

    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(X, y, cat_features=cat_feats)
    model.save_model(MODEL_PATH)
    with open(GLOBAL_FREQ_PATH, 'wb') as f:
        pickle.dump(global_freq, f)
    st.cache_resource.clear()
    return model, global_freq

# ========================
# Функция предсказания (модель + правила)
# ========================
def predict_next_ml(context_row, model, global_freq, history):
    """Возвращает оптимальный ход (букву 'К','Н','Б') и уверенность."""
    df_input = pd.DataFrame([context_row])
    df_feat = create_features(df_input, global_freq)
    X, cat_feats = prepare_features(df_feat)
    # Приводим колонки к тем, что были при обучении (на случай расхождений)
    # model.feature_names_ может не совпадать, поэтому переиндексируем
    expected_features = model.feature_names_
    if expected_features is not None:
        X = X.reindex(columns=expected_features, fill_value=0)
    proba = model.predict_proba(X)[0]   # вероятности классов 0,1,2
    # В обучении: optimal_idx = 0->Б, 1->К, 2->Н (map {'К':1, 'Н':2, 'Б':0})
    idx_to_move = {0: 'Б', 1: 'К', 2: 'Н'}
    pred_idx = np.argmax(proba)
    pred_move = idx_to_move[pred_idx]
    confidence = proba[pred_idx]

    # Правила по текущему матчу
    if history:
        last = history[-1]
        last_opp = last['opp_move']
        last_outcome = last['outcome']
        # После ничьей – бей ход соперника
        if last_outcome == 'draw' and last_opp in ['К','Н','Б']:
            beat = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
            return beat[last_opp], 1.0   # полная уверенность
        # Дважды повтор – избегаем бить этот ход
        if len(history) >= 2:
            prev_opp = history[-2]['opp_move']
            if last_opp == prev_opp and last_opp in ['К','Н','Б']:
                beat = {'К': 'Н', 'Н': 'Б', 'Б': 'К'}
                avoid = beat[last_opp]
                alternatives = [m for m in ['К','Н','Б'] if m != avoid]
                if pred_move == avoid:
                    pred_move = alternatives[0]
                    # Уверенность можно оставить от модели (или снизить)
    return pred_move, confidence

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
    if 'model' not in st.session_state:
        model, global_freq = load_or_train_model()
        st.session_state.model = model
        st.session_state.global_freq = global_freq

# ========================
# Основное приложение
# ========================
st.set_page_config(page_title="Помощник в КНБ (ML)", layout="wide")
st.title("🎮 Помощник 'Камень-Ножницы-Бумага'")

ensure_csv()
if 'clean_done' not in st.session_state:
    clean_unfinished()
    st.session_state.clean_done = True
init_session()

# Сайдбар с кнопкой переобучения
with st.sidebar:
    st.header("⚙️ Модель")
    if st.button("♻️ Переобучить на всех данных"):
        with st.spinner("Обучение..."):
            model, global_freq = retrain_model()
            st.session_state.model = model
            st.session_state.global_freq = global_freq
        st.success("Модель обновлена!")

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

            # Предсказание первого раунда
            dummy_row = {
                'round': 1,
                'player_name': st.session_state.player_name,
                'win_category': compute_win_category(wins),
                'opp_match_wins': wins,
                'opp_match_winrate': winrate,
                'stake': stake,
                'score_me_before': 0,
                'score_opp_before': 0,
                'streak_draws': 0,
                'is_last_round': 0,
            }
            for i in range(1,7):
                dummy_row[f'prev{i}_opp_move'] = '-1'
                dummy_row[f'prev{i}_my_move'] = '-1'
                dummy_row[f'prev{i}_outcome'] = 'none'

            opt, conf = predict_next_ml(dummy_row, st.session_state.model,
                                        st.session_state.global_freq, [])
            st.session_state.next_prediction = (LETTER_TO_MOVE[opt], opt, conf, 0)
            st.rerun()

# -------------------- Игровой процесс --------------------
elif st.session_state.game_state == 'playing':
    st.info(f"Счёт: **{st.session_state.score_me} : {st.session_state.score_opp}** | "
            f"Раунд {st.session_state.round_num} | Матч #{st.session_state.match_id} | "
            f"Противник: {st.session_state.player_name}")

    if st.session_state.next_prediction:
        your_move, probable_opp, confidence, support = st.session_state.next_prediction
        msg = f"🤖 Рекомендация на **раунд {st.session_state.round_num}**: **{your_move}**"
        msg += f"\n\n📈 Уверенность модели: {confidence:.1%}"
        st.success(msg)

    st.subheader("Выберите ход противника и исход раунда")
    col1, col2, col3 = st.columns(3)
    opp_type_n = "primary" if st.session_state.selected_opp == "Ножницы" else "secondary"
    opp_type_k = "primary" if st.session_state.selected_opp == "Камень" else "secondary"
    opp_type_b = "primary" if st.session_state.selected_opp == "Бумага" else "secondary"

    with col1:
        if st.button("✌️ Ножницы", key="opp_n", use_container_width=True, type=opp_type_n):
            st.session_state.selected_opp = "Ножницы"; st.rerun()
    with col2:
        if st.button("✊ Камень", key="opp_k", use_container_width=True, type=opp_type_k):
            st.session_state.selected_opp = "Камень"; st.rerun()
    with col3:
        if st.button("✋ Бумага", key="opp_b", use_container_width=True, type=opp_type_b):
            st.session_state.selected_opp = "Бумага"; st.rerun()

    st.markdown("**Исход для вас:**")
    col4, col5, col6 = st.columns(3)
    out_type_l = "primary" if st.session_state.selected_outcome == "Поражение" else "secondary"
    out_type_d = "primary" if st.session_state.selected_outcome == "Ничья" else "secondary"
    out_type_w = "primary" if st.session_state.selected_outcome == "Победа" else "secondary"

    with col4:
        if st.button("😞 Поражение", key="out_l", use_container_width=True, type=out_type_l):
            st.session_state.selected_outcome = "Поражение"; st.rerun()
    with col5:
        if st.button("🤝 Ничья", key="out_d", use_container_width=True, type=out_type_d):
            st.session_state.selected_outcome = "Ничья"; st.rerun()
    with col6:
        if st.button("😊 Победа", key="out_w", use_container_width=True, type=out_type_w):
            st.session_state.selected_outcome = "Победа"; st.rerun()

    next_disabled = (st.session_state.selected_opp is None or st.session_state.selected_outcome is None)
    if st.button("➡️ Записать раунд и предсказать следующий", use_container_width=True, disabled=next_disabled):
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

        if outcome == 'win':
            st.session_state.score_me += 1
            st.session_state.streak_draws = 0
        elif outcome == 'lose':
            st.session_state.score_opp += 1
            st.session_state.streak_draws = 0
        else:
            st.session_state.streak_draws += 1

        # Сохранение в CSV
        df_new = pd.DataFrame([new_row])
        if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
            df_new.to_csv(DATA_PATH, index=False)
        else:
            df_new.to_csv(DATA_PATH, mode='a', header=False, index=False)
        st.cache_data.clear()
        time.sleep(0.05)

        if st.session_state.score_me >= 3 or st.session_state.score_opp >= 3:
            st.session_state.game_state = 'finished'
            st.session_state.next_prediction = None
            clean_unfinished()
            st.success(f"🏆 Матч #{st.session_state.match_id} окончен! Счёт {st.session_state.score_me}:{st.session_state.score_opp}")
            st.rerun()

        # Предсказание следующего раунда
        next_round = st.session_state.round_num + 1
        ctx = {
            'round': next_round,
            'player_name': st.session_state.player_name,
            'win_category': win_cat,
            'opp_match_wins': st.session_state.opp_stats['wins'],
            'opp_match_winrate': st.session_state.opp_stats['winrate'],
            'stake': st.session_state.opp_stats['stake'],
            'score_me_before': st.session_state.score_me,
            'score_opp_before': st.session_state.score_opp,
            'streak_draws': st.session_state.streak_draws,
            'is_last_round': 1 if (st.session_state.score_me == 2 or st.session_state.score_opp == 2) else 0
        }
        for i in range(1,7):
            if len(st.session_state.history) >= i:
                rec = st.session_state.history[-i]
                ctx[f'prev{i}_opp_move'] = rec['opp_move']
                ctx[f'prev{i}_my_move'] = rec['my_move']
                ctx[f'prev{i}_outcome'] = rec['outcome']
            else:
                ctx[f'prev{i}_opp_move'] = '-1'
                ctx[f'prev{i}_my_move'] = '-1'
                ctx[f'prev{i}_outcome'] = 'none'

        opt_move, conf = predict_next_ml(ctx, st.session_state.model,
                                         st.session_state.global_freq,
                                         st.session_state.history)
        st.session_state.next_prediction = (LETTER_TO_MOVE[opt_move], opt_move, conf, 0)
        st.session_state.round_num = next_round
        st.session_state.selected_opp = None
        st.session_state.selected_outcome = None
        st.rerun()

    # Боковые панели (история матча и последние записи)
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
            show_cols = ['match_id', 'round', 'player_name', 'win_category',
                         'opp_move', 'my_move', 'outcome',
                         'score_me_before', 'score_opp_before']
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
    st.info(f"Итоговый счёт матча #{st.session_state.match_id}: {st.session_state.score_me} : {st.session_state.score_opp} | "
            f"Противник: {st.session_state.player_name}")
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