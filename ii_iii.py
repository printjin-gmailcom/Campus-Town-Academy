import chromedriver_autoinstaller
from selenium import webdriver

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

url = 'https://news.v.daum.net/v/20160312084803770'
browser.get(url)

from bs4 import BeautifulSoup
soup = BeautifulSoup( browser.page_source, 'html.parser' )

soup

# <h3 class="tit_view" data-translation="true">[이세돌 vs 알파고 3국] 난공불락 알파고..'이세돌다운 수'가 실낱 희망</h3>
soup.select('h3')

# <h3 class="tit_view" data-translation="true">[이세돌 vs 알파고 3국] 난공불락 알파고..'이세돌다운 수'가 실낱 희망</h3>
title = soup.select('h3')[0].text
print(title)

# <h3 class="tit_view" data-translation="true">[이세돌 vs 알파고 3국] 난공불락 알파고..'이세돌다운 수'가 실낱 희망</h3>
tag_list = soup.select('h3')
print(len(tag_list))

tag_list = soup.select('h3.tit_view')
len(tag_list)

# tag_list = soup.select('태그정보')
# tag_list = soup.select('태그명')
# tag_list = soup.select(.class속성값)
# tag_list = soup.select(#id속성값)
# tag_list = soup.select(부모태그정보 > 자식태그정보)

tag_list = soup.select('h3')
tag = tag_list[0]
tag.text

title = soup.select('h3.tit_view')[0].text
print(title)

title = soup.select('h3.tit_view')[0]
title

company = soup.select('img')
len(company)

company = soup.select(' a > img.thumb_g')
len(company)

company = soup.select(' a.link_cp > img.thumb_g')
len(company)

