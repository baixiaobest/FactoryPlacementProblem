from selenium import webdriver
from selenium.webdriver.common.by import By
import pickle
import re

def get_population1():
    name_population_dict = {}
    # selenium爬取省份名称
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('headless')
    print("creating driver")
    driver = webdriver.Chrome(options=chrome_options)
    print("Driver created")
    url = 'https://zh.wikipedia.org/zh-cn/中華人民共和國城市城區人口排名'
    driver.get(url)
    # 需要点击才显示出全部省份列表
    entries = driver.find_elements(By.CSS_SELECTOR, ".wikitable tr")
    print("results acquired")
    entries.pop(0)
    for entry in entries:
        name_ele = entry.find_elements(By.CSS_SELECTOR, 'td')[1]
        population_ele = entry.find_elements(By.CSS_SELECTOR, 'td')[4]
        name_population_dict[name_ele.text] = float(population_ele.text)
    driver.close()
    print("driver closed")

    print(name_population_dict)

    filehandler = open('data/populationdata', 'wb')
    pickle.dump(name_population_dict, filehandler)
    filehandler.close()

def get_population2():
    name_population_dict = {}
    # selenium爬取省份名称
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('headless')
    print("creating driver")
    driver = webdriver.Chrome(options=chrome_options)
    print("Driver created")
    url = 'https://baike.baidu.com/item/中国城市人口排名表/16620508'
    driver.get(url)
    tables = driver.find_elements(By.CSS_SELECTOR, "table")
    tables = tables[0:7]
    print("results acquired")
    for table in tables:
        for entry in table.find_elements(By.CSS_SELECTOR, 'tr'):
            cells = entry.find_elements(By.CSS_SELECTOR, 'td')
            order = re.sub("[^0-9,.]", '', cells[0].text)
            if order == '':
                continue
            name = cells[1].text
            population = re.sub("[^0-9,.]", '', cells[2].text)
            name_population_dict[name] = float(population)
    driver.close()
    print("driver closed")

    print(name_population_dict)

    filehandler = open('data/populationdata', 'wb')
    pickle.dump(name_population_dict, filehandler)
    filehandler.close()

if __name__=="__main__":
    get_population2()
