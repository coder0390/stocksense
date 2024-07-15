import json
import os

from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render

from news.api.word_cloud import generate_wordcloud
from utils.database import select_all_news, select_news_by_keyword

# Create your views here.


database_path = "stocks421.db"
store_path = "pic"


def all_news(request):
    data = json.loads(request.body)
    page_size = data.get('page_size')
    page_no = data.get('page_no')

    news, total_pages = select_all_news(database_path, page_size, page_no)

    response_data = {
        'news': news.to_dict('records'),
        'total_pages': total_pages
    }

    return JsonResponse(response_data, safe=False)


def news_by_keyword(request):
    data = json.loads(request.body)
    keyword = data.get('keyword')
    page_size = data.get('page_size')
    page_no = data.get('page_no')

    # print(keyword, page_size, page_no)

    news, total_pages = select_news_by_keyword(database_path, keyword, page_size, page_no)

    response_data = {
        'news': news.to_dict('records'),
        'total_pages': total_pages
    }

    return JsonResponse(response_data, safe=False)


def word_cloud(request):
    data = json.loads(request.body)
    id_ = data.get('id')

    current_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(os.path.dirname(current_path))
    upload_dir = os.path.join(project_root_path, store_path)
    wordcloud = generate_wordcloud(database_path, id_, upload_dir)

    wordcloud_url = os.path.join(settings.MEDIA_URL, os.path.basename(wordcloud))

    return JsonResponse({'url': 'http://127.0.0.1:8000' + wordcloud_url})
