import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd

# Set up Selenium options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run in headless mode
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

# Set up the Chrome driver
#NOTE: YOU HAVE TO DOWNLOAD GOOGLE CHROMEDRIVER AND SET IT UP.
service = Service("D:\\False C Drive\\chrome-win64\\chromedriver.exe")  # Update path to ChromeDriver according to your local machine
driver = webdriver.Chrome(service=service, options=chrome_options)

# IMDb URL
#NOTE: The url currently points to latest movies. Update the url to get the desired year.
#How to update: paste the url link in a browser and modify the criteria as desired (for example the date range). Then copy paste the new url here. 
url = "https://www.imdb.com/search/title/?title_type=feature&release_date=2021-01-01,2025-01-31&num_votes=500,&sort=year,asc" #1992 jan-1993 dec
driver.get(url)
time.sleep(5)

# Click the "See More" button multiple times to load more movies
i=0
while i<50000:
    try:
        button = driver.find_element(By.CLASS_NAME, "ipc-see-more__text")
        ActionChains(driver).move_to_element(button).click().perform()
        i+=50
        time.sleep(3)
    except:
        break  # Exit loop if button not found

# Get page source and parse with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Extract movie details
movies = []
for item in soup.select("li.ipc-metadata-list-summary-item"):
    title = item.select_one("h3").text if item.select_one("h3") else "N/A"
    year = item.select_one("span.sc-d5ea4b9d-7:nth-of-type(1)").text if item.select_one(
        "span.sc-d5ea4b9d-7:nth-of-type(1)") else "N/A"
    rating = item.select_one("span.ipc-rating-star--rating").text if item.select_one(
        "span.ipc-rating-star--rating") else "N/A"
    num_votes = item.select_one("span.ipc-rating-star--voteCount").text if item.select_one(
        "span.ipc-rating-star--voteCount") else "N/A"
    synopsis = item.select_one("div.ipc-html-content-inner-div").text if item.select_one(
        "div.ipc-html-content-inner-div") else "N/A"
    runtime = item.select_one("span.sc-d5ea4b9d-7:nth-of-type(2)").text if item.select_one(
        "span.sc-d5ea4b9d-7:nth-of-type(2)") else "N/A"

    movies.append([title, year, rating, num_votes, synopsis, runtime])

# Convert to DataFrame
df = pd.DataFrame(movies, columns=["Title", "Year", "Rating", "Votes", "Synopsis", "Runtime"])

# Save to CSV
#IMPORTANT: Make sure to update the file name each time or previous data shall be lost. 
filename = "imdb_2021-2024_movies_filtered.csv"
df.to_csv(filename, index=False)

# Close driver
driver.quit()

print(f"Scraping complete. Data saved to {filename}")
