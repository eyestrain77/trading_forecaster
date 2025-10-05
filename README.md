# SeedGuessr Forecast
### Для запуска
1. Положите данные в папку Data и пропишите пути к ним в файле data_processing.py в функции load_data. Также при необходимости замените пути в news_encoder_binary.py (по дефолту data/candles.csv, data/candles_2.csv), а также залейте news.csv локально так как он слишком большой чтобы добавить его в гитхаб
2. pip install requirements.txt
3. Запустите news_encoder_binary.py
4. Запустите main.py
5. main.py вернет предикты в файле submission.csv
