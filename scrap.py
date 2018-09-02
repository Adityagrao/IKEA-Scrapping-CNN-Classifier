import urllib.request
from bs4 import BeautifulSoup
import argparse
import cv2
import os
import numpy as np
import posixpath
from tqdm import tqdm
args = argparse.ArgumentParser(description="Add Categories")
args.add_argument('-d', '--dmp', nargs='+', type=str, dest='list', required=True)
BASE_URL = "https://www.ikea.com/us/en/catalog/allproducts/department/"
BASE_WEB = "https://www.ikea.com"
CATEGORIES = args.parse_args().list
IMAGE_DIR = "images/"
urls = []
sauce = urllib.request.urlopen(BASE_URL).read()
soup = BeautifulSoup(sauce, 'lxml')
for CATEGORY in tqdm(CATEGORIES):
    if not os.path.exists(IMAGE_DIR + CATEGORY):
        os.makedirs(IMAGE_DIR + CATEGORY)
        print("DIR ", CATEGORY + " Created")
    for url in soup.find_all("a", text=lambda text: text and CATEGORY in text):
        child = url.get('href')
        child_sauce = urllib.request.urlopen(BASE_WEB + child).read()
        child_soup = BeautifulSoup(child_sauce, 'lxml')
        print("Downloading Images for ", CATEGORY)
        for img in tqdm(child_soup.find_all("img")):
            if ".JPG" in str(img.get('src')):
                IMAGE_URL = BASE_WEB + str(img.get('src'))
                request = urllib.request.Request(IMAGE_URL)
                response = urllib.request.urlopen(request)
                binary_str = response.read()
                byte_array = bytearray(binary_str)
                numpy_array = np.asarray(byte_array, dtype="uint8")
                image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
                ID = posixpath.basename(IMAGE_URL)
                file_name = IMAGE_DIR + CATEGORY + "/" + ID
                cv2.imwrite(file_name, image)
print("Done")
