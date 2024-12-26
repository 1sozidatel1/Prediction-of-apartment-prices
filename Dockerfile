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

