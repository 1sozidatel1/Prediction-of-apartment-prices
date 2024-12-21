import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
from datasets import load_dataset

model = tf.keras.models.load_model('model/price_model.h5')
location_mapping = {1: 'Астрахань', 2: 'Волгоград', 3: 'Краснодар', 4: 'Ростов-на-Дону', 5: 'Майкоп', 6: 'Элиста'}
district_mapping = {11: 'Астрахань - Кировский', 12: 'Астрахань - Ленинский', 13: 'Астрахань - Советский', 14: 'Астрахань - Трусовский', 15: 'Астрахань - Центральный',
                    21: 'Волгоград - Ворошиловский', 22: 'Волгоград - Дзержинский', 23: 'Волгоград - Кировский', 24: 'Волгоград - Красноармейский', 25: 'Волгоград - Краснооктябрьский', 26: 'Волгоград - Советский', 27: 'Волгоград - Тракторозаводский', 28: 'Волгоград - Центральный',
                    31: 'Краснодар - Западный', 32: 'Краснодар - Карасунский', 33: 'Краснодар - Прикубанский', 34: 'Краснодар - Центральный',
                    41: 'Ростов-на-Дону - Ворошиловский', 42: 'Ростов-на-Дону - Железнодорожный', 43: 'Ростов-на-Дону - Кировский', 44: 'Ростов-на-Дону - Ленинский', 45: 'Ростов-на-Дону - Октябрьский', 46: 'Ростов-на-Дону - Первомайский', 47: 'Ростов-на-Дону - Пролетарский', 48: 'Ростов-на-Дону - Советский',
                    51: 'Майкоп - Центральный',
                    61: 'Элиста - Центральный', }
author_type_mapping = {0: 'Риелтор', 1: 'Частный владелец'}
dataset = load_dataset("1sozidatel1/sfd_housing_prices_november_2024")
data = pd.DataFrame(dataset['train'])
feature_cols = ['author_type_id', 'location_id', 'district_id', 'floor', 'rooms_count', 'total_meters']

X = data[feature_cols]
X_min = X.min()
X_max = X.max()
y = data['price'].values.reshape(-1, 1)
y_min = np.min(y)
y_max = np.max(y)

def predict_price(author_type, location, district, floor, rooms_count, total_meters):
    author_type_id = list(author_type_mapping.keys())[list(author_type_mapping.values()).index(author_type)]
    location_id = list(location_mapping.keys())[list(location_mapping.values()).index(location)]
    district_id = list(district_mapping.keys())[list(district_mapping.values()).index(district)]
    input_data = np.array([[author_type_id, location_id, district_id, floor, rooms_count, total_meters]])
    input_data_normalized = (input_data - X_min.values) / (X_max.values - X_min.values)
    prediction = model.predict(input_data_normalized)
    prediction_unscaled = prediction * (y_max - y_min) + y_min
    return np.round(prediction_unscaled.flatten()[0], 1)

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
iface.launch()
