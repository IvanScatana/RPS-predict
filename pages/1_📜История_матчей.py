import streamlit as st
import pandas as pd
import os

# ========================
# Конфигурация
# ========================
DATA_PATH = 'rps_data.csv'

# Словари для перевода в русские названия (для читаемости)
MOVE_TO_LETTER = {"Камень": "К", "Ножницы": "Н", "Бумага": "Б"}
LETTER_TO_MOVE = {v: k for k, v in MOVE_TO_LETTER.items()}
OUTCOME_TO_EN = {"Победа": "win", "Поражение": "lose", "Ничья": "draw"}
EN_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_EN.items()}

st.set_page_config(page_title="История матчей", layout="wide")
st.title("📜 История сохранённых раундов (завершённые матчи)")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
    if not df.empty:
        # Переводим в читаемый вид
        df_disp = df.copy()
        for col in ['opp_move', 'my_move', 'prev_opp_move', 'prev2_opp_move']:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].map(lambda x: LETTER_TO_MOVE.get(x, x))
        if 'outcome' in df_disp.columns:
            df_disp['outcome'] = df_disp['outcome'].map(EN_TO_OUTCOME)
        if 'prev_outcome' in df_disp.columns:
            df_disp['prev_outcome'] = df_disp['prev_outcome'].map(lambda x: EN_TO_OUTCOME.get(x, x))
        
        # Отображаем большую таблицу на всю ширину
        st.dataframe(df_disp, use_container_width=True, height=600)
        
        # Кнопка скачивания CSV
        csv_data = df.to_csv(index=False, sep=',', encoding='utf-8')
        st.download_button(
            label="💾 Скачать CSV (сырые данные)",
            data=csv_data,
            file_name="rps_data.csv",
            mime="text/csv"
        )
    else:
        st.write("Файл истории пуст. Сыграйте несколько матчей в основном приложении.")
else:
    st.write("Файл истории не найден. Сначала запустите основное приложение и сохраните хотя бы один раунд.")