FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
# Переходим в папку struct и запускаем start.py
WORKDIR /app/struct
CMD ["python", "start.py"]

# docker build -t danems/prediction_app .
# docker run -it --rm danems/prediction_app
# или
# docker run -it -p 6000:6000 danems/prediction_app


# Коомпиляция под amd и загрузка
# FROM ubuntu:20.04 - мейби пригодится для коомпиляции
# Создаем и активируем билдер
#docker buildx create --name mybuilder --use
#docker buildx inspect --bootstrap

# Строим образ для архитектуры AMD
#docker buildx build --platform linux/amd64 -t danems/prediction_app:latest .

# Логинимся в Docker Hub
#docker login

# Отправляем образ на Docker Hub
#docker push danems/prediction_app:latest

# Запускаем контейнер (если у тебя есть Docker для AMD)
#docker run -it --rm danems/prediction_app:latest
# Или
#docker run -it -p 6000:6000 danems/prediction_app:latest

