import chromedriver_autoinstaller
from selenium import webdriver
from bs4 import BeautifulSoup

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

url = 'https://naver.com'
browser.get(url)

# soup = BeautifulSoup( browser.page_source. 'html.parser')
# soup.select('태그조건')

# 검색어 입력칸 선택
# browser.find_elements('어떤기준으로', '찾고싶은조건')
input_search = browser.find_elements('css selector', 'input.input_text')[0]
input_search.clear()
input_search.send_keys('부동산')

import time

word_list = ['노트북 거치대', '모니터', '마우스']
for word in word_list:
    input_search = browser.find_elements('css selector', 'input.input_text')[0]
    input_search.clear()
    input_search.send_keys(word)
    time.sleep(0.5)

btn_search = browser.find_elements('css selector', 'span.ico_search_submit')[0]
btn_search.click()
