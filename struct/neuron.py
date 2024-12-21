import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Загрузка датасета 'sfd_housing_prices_november_2024'
dataset = load_dataset("1sozidatel1/sfd_housing_prices_november_2024")
data = pd.DataFrame(dataset['train'])

# Маппинг
location_mapping = {1: 'Астрахань', 2: 'Волгоград', 3: 'Краснодар', 4: 'Ростов-на-Дону', 5: 'Майкоп', 6: 'Элиста'}
district_mapping = {11: 'Астрахань - Кировский', 12: 'Астрахань - Ленинский', 13: 'Астрахань - Советский', 14: 'Астрахань - Трусовский', 15: 'Астрахань - Центральный',
                    21: 'Волгоград - Ворошиловский', 22: 'Волгоград - Дзержинский', 23: 'Волгоград - Кировский', 24: 'Волгоград - Красноармейский', 25: 'Волгоград - Краснооктябрьский', 26: 'Волгоград - Советский', 27: 'Волгоград - Тракторозаводский', 28: 'Волгоград - Центральный',
                    31: 'Краснодар - Западный', 32: 'Краснодар - Карасунский', 33: 'Краснодар - Прикубанский', 34: 'Краснодар - Центральный',
                    41: 'Ростов-на-Дону - Ворошиловский', 42: 'Ростов-на-Дону - Железнодорожный', 43: 'Ростов-на-Дону - Кировский', 44: 'Ростов-на-Дону - Ленинский', 45: 'Ростов-на-Дону - Октябрьский', 46: 'Ростов-на-Дону - Первомайский', 47: 'Ростов-на-Дону - Пролетарский', 48: 'Ростов-на-Дону - Советский',
                    51: 'Майкоп - Центральный',
                    61: 'Элиста - Центральный', }
author_type_mapping = {0: 'Риелтор', 1: 'Частный владелец'}

# Предобработка данных
feature_cols = ['author_type_id', 'location_id', 'district_id', 'floor', 'rooms_count', 'total_meters']
target_col = 'price'

# Разделение данных на X и y
X = data[feature_cols]
y = data[target_col].values.reshape(-1, 1)

# Нормализация данных (min-max scaling)
X_min = X.min()
X_max = X.max()
X_normalized = (X - X_min) / (X_max - X_min)
y_min = np.min(y)
y_max = np.max(y)
y_normalized = (y - y_min) / (y_max - y_min)

# Разделение на тренировочную и тестовую выборки (с учетом стратификации)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, stratify=data['location_id'], random_state=42)

# Создание нейросетевой модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Компиляция модели с использованием Huber Loss
model.compile(optimizer='adam', loss=tf.keras.losses.Huber())

# Создание обратного вызова для ранней остановки
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение модели с использованием ранней остановки
model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

# Функция для предсказания цены
def predict_price(author_type, location, district, floor, rooms_count, total_meters):
    author_type_id = list(author_type_mapping.keys())[list(author_type_mapping.values()).index(author_type)]
    location_id = list(location_mapping.keys())[list(location_mapping.values()).index(location)]
    district_id = list(district_mapping.keys())[list(district_mapping.values()).index(district)]
    input_data = np.array([[author_type_id, location_id, district_id, floor, rooms_count, total_meters]])
    input_data_normalized = (input_data - X_min.values) / (X_max.values - X_min.values)
    prediction = model.predict(input_data_normalized)
    prediction_unscaled = prediction * (y_max - y_min) + y_min
    return np.round(prediction_unscaled.flatten()[0], 1)

# Создание интерфейса с помощью Gradio
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(choices=list(author_type_mapping.values()), label="Тип продавца"),
        gr.Dropdown(choices=list(location_mapping.values()), label="Город"),
        gr.Dropdown(choices=list(district_mapping.values()), label="Район"),
        gr.Number(label="Этаж", minimum=1, maximum=50),  # Установка ограничения на этаж
        gr.Dropdown(choices=[1, 2, 3, 4, 5], label="Количество комнат"),  # Изменено на Dropdown
        gr.Number(label="Площадь, кв.м")
    ],
    outputs=gr.Number(label="Предсказанная цена, (тыс. руб.)")
)

# Запуск интерфейса
iface.launch()
