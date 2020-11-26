import selenium
from selenium import webdriver as wb
import urllib.request
# Install selenium using pip install selenium
# chromedriver_path : path to chromedriver.exe
# Download from : https://chromedriver.chromium.org/

chromedriver_path = '..path/../chromedriver'

# source : path to webpage source

source = 'https://www.nykaa.com/skin/cleansers/face-wash/c/8379?root=nav_3&page_no=29'

webD = wb.Chrome(executable_path=chromedriver_path)
webD.get(source)
productInfoList = webD.find_elements_by_class_name("card-img")


list_of_links = []
for element in productInfoList:
    div_tag = element.find_element_by_tag_name('div')

    list_of_links.append(
        div_tag.find_element_by_tag_name('img').get_property('src'))

# print(list_of_links)

formatted_links = []
for link in list_of_links:
    new_link = link.replace('w-276,h-276', 'w-800,h-800')
    formatted_links.append(new_link)


# output_path : Where downloaded images will be stored.Add a "/" at the end
output_path = ""

for link in formatted_links:
    urllib.request.urlretrieve(link, output_path + link.split('/')[-1])
