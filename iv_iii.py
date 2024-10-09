import chromedriver_autoinstaller
from selenium import webdriver
from bs4 import BeautifulSoup

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

url = 'https://youtube-rank.com/board/bbs/board.php?bo_table=youtube&page=1'
browser.get(url)

browser.page_source

BeautifulSoup(browser.page_source, "html.parser")

soup = BeautifulSoup(browser.page_source, "html.parser")

channel_list = soup.select('tr')[1:-1]
len(channel_list)

print(channel_list[-2].text)

channel_list = soup.select('form > table > tbody > tr')
len(channel_list)

channel_list = soup.select('tr.aos-init')
len(channel_list)

