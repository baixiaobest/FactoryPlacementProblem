from selenium import webdriver
from selenium.webdriver.common.by import By
import pickle

if __name__=="__main__":
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
