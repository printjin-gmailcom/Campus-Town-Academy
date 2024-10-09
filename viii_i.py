# 1. 크롬브라우저 열기
import chromedriver_autoinstaller
from selenium import webdriver
from bs4 import BeautifulSoup

# chromedriver 최신버전설치
chromedriver_autoinstaller.install()
# 브라우저 열기
browser = webdriver.Chrome()

url = 'https://search.shopping.naver.com/catalog/25995128523?cat_id=50002543'
browser.get(url)

soup = BeautifulSoup(browser.page_source, "html.parser")

item_list = soup.select('ul.productList_list_seller__XGhCk > li')
len(item_list)

<a href = 'URL'>
tag = '속성명'

for item in item_list:
    title = item.select('a.productList_title__R1qZP')[0].text
    url = item.select('a.productList_title__R1qZP')[0]['href']
    try:
        mall = item.select('img')[0]['alt']
    except:
        mall = item.select('a.productList_mall_link__TrYxC > span')[0].text
    price = item.select('a > span > em')[0].text
    deli = item.select('div.productList_delivery__WwSwL')[0].text
    if deli == '무료배송':
        deli_num = '0'
    elif deli == '착불':
        deli_num = '5,000'
    else:
        deli_num = deli

    print(title, mall,price,deli)



browser.find_elements("css selector", "a")

page_button_list = browser.find_elements("css selector", "div.productList_seller_wrap__FZtUS > div.pagination_pagination__JW7zT > a")
for page_button in page_button_list:
    page_button.click()
    time.sleep(1)

