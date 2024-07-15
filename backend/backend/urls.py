"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from news.views import all_news, news_by_keyword, word_cloud
from prediction.views import get_predict
from strategy.views import hello_world, stock_codes, stock_data, stocks_by_date, strategy_data, generate_weights

urlpatterns = [
    path("admin/", admin.site.urls),
    path("hello_world/", hello_world),
    path("database/stock_codes/", stock_codes),
    path('database/stock_data/', stock_data),
    path('database/date_data/', stocks_by_date),
    path('strategy/data/', strategy_data),
    path('strategy/weights/', generate_weights),
    path('prediction/', get_predict),
    path('news/all/', all_news),
    path('news/keyword/', news_by_keyword),
    path('news/wordcloud/', word_cloud)
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
