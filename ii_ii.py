# 라이브러리 설치
! pip install chromedriver-autoinstaller
! pip install selenium

import chromedriver_autoinstaller
from selenium import webdriver

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

word_list = ['가', '나', '다']
for word in word_list:
    url = f'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query={word}'
    browser.get(url)

