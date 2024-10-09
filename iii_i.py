import chromedriver_autoinstaller
from selenium import webdriver
from bs4 import BeautifulSoup

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

url = 'https://youtube-rank.com/board/bbs/board.php?bo_table=youtube&page=1'
browser.get(url)

soup = BeautifulSoup(browser.page_source, "html.parser")

soup = BeautifulSoup(browser.page_source, "html.parser")

channel_list = soup.select('tr.aos-init')
len(channel_list)

channel = channel_list[0]
channel

title = channel.select("h1>a")[0].text.strip()
print(title)

for channel in channel_list:
    title = channel.select("h1>a")[0].text.strip()
    print(title)

channel= channel_list[0]
category = channel.select("p.category")[0].text.strip()
category

for channel in channel_list:
    category = channel.select("p.category")[0].text.strip()
    print(category)

for channel in channel_list:
    sub = channel.select("td.subscriber_cnt")[0].text.strip()
    view = channel.select("td.view_cnt")[0].text.strip()
    video = channel.select("td.video_cnt")[0].text.strip()
    print(sub,view,video)



