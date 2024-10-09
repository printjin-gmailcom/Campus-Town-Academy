import 라이브러리명
from 라이브러리명 import 명령어

browser.get(url)

browser.page_source

soup = BeautifulSoup(데이터, 'htmi.parser')
soup = BeautifulSoup(browser.page_source, 'htmi.parser')

soup.select('태그정보')
soup.select('태그명')
soup.select('class 속성값')
soup.select('#id속성값')
soup.select('부모태그정보>지식태그정보')

tag_list = soup.select()



<태그> <태그>
<태그명> </태그명>
<태그ㅁ
