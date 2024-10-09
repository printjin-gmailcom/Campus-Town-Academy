import chromedriver_autoinstaller
from selenium import webdriver
from bs4 import BeautifulSoup

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

url = 'https://naver.com'
browser.get(url)

import time

from datetime import datetime

datetime.now().isoformat()[:10]


import pandas as pd

word_list = ['파이썬', '부동산', '주식', '코인', '취업', '노트북' ]
for word in word_list:
    results = []
    input_search = browser.find_elements('css selector', 'input.input_text')[0]
    input_search.clear()
    input_search.send_keys(word)
    time.sleep(0.5)

    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')

    related_list = soup.select('span.fix')[1:]
    for related in related_list:
        data = [word, related.text, datetime.now().isoformat()[:10]]
        results.append(data)


    df = pd.DataFrame(results)
    df.columns = ['검색어','연관검색어', '검색일자']
    df.to_excel(f'./연관검색어_{datetime.now().isoformat()[:10]}_{word}.xlsx', index = False)

