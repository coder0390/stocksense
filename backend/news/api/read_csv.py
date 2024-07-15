import csv
from datetime import datetime

from utils.database import insert_news_data


def read_csv(csv_path, database_path):
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print(reader.fieldnames)
        for row in reader:
            url = row['\ufeffurl']
            overview = row['overview']
            title = row['title']
            writer = row['writer']

            time_str = row['time'].replace('Published: ', '').strip()
            try:
                time = datetime.strptime(time_str, '%I:%M%p, %d %b %Y')
            except ValueError:
                # print(url)
                time = None

            content = row['content']

            try:
                if int(row['predict']) == 1:
                    sentiment = 'positive'
                else:
                    sentiment = 'negative'
            except (ValueError, KeyError):
                # print(title)
                sentiment = 'unknown'

            print(url)
            # insert_news_data(database_path, url, overview, title, writer, time, content, sentiment)


csv_path = "../../news.csv"
database_path = "../../stocks421.db"
read_csv(csv_path, database_path)
