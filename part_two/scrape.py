"""
    Original code Web Scraping British Airways Reviews data from Kaggle 
    https://www.kaggle.com/code/praveensaik/web-scraping-british-airways-reviews-data
"""

import pandas as pd

import re
from bs4 import BeautifulSoup
import requests
from utils import prepare

import warnings
warnings.filterwarnings('ignore')

def scraper(url, file_name,pages= 250):
   
    reviews = []
    date = []
    country = []
    type_of_traveller = []
    seat_type=[]
    route=[]
    recommended=[]

    for i in range(1, pages + 1):
        print(f"Scraping page {i}")

        page = requests.get(f"{url}/page/{i}/")

        soup = BeautifulSoup(page.content, "html.parser")

        # Reviews
        for item in soup.find_all("div", class_="text_content"):
            text = re.sub(r'Trip Verified ', '', item.text)
            text = re.sub(r'Not Verified ', '', text)
            reviews.append(text)

        # Date
        for item in soup.find_all("time"):
            date.append(item.text)

        # Country
        for item in soup.find_all("h3"):
            country_text = item.text.strip()
            # Use regular expression to extract the country name between parentheses
            country_name = re.search(r'\((.*?)\)', country_text)[1]
            country.append(country_name)

        # Seat Type
        for item in soup.find_all('td', class_='review-rating-header cabin_flown'):
            seat_type_text = item.find_next_sibling('td').text.strip()
            seat_type.append(seat_type_text)

        # Recommended
        for item in soup.find_all('td', class_='review-rating-header recommended'):
            recommended_text = item.find_next_sibling('td').text.strip()
            recommended.append(recommended_text)


        print(f" -----> {len(reviews)} total reviews")

    # Check if all lists have the same length
    if len(reviews) == len(date) == len(country) == len(seat_type)== len(recommended):

        # Create a DataFrame using dictionary
        df = pd.DataFrame({
            'reviews': reviews,
            'date': date,
            'country': country,
            'seat_type':seat_type,
            'recommended': recommended
        })
        print("DataFrame created Successfully!")
    else:
        print('Error: Lists have different lengths')
        print(f"length of reviews: {len(reviews)}")
        print(f"length of date: {len(date)}")
        print(f"length of country: {len(country)}")
        print(f"length of recommended: {len(recommended)}")
        print(f"length of seat_type: {len(seat_type)}")


    stars=[]
    for _ in range(1,pages+1):
        # Stars
        for item in soup.find_all("div", class_='rating-10'):
            star_rating = item.span.text.strip() if item.span else "N/A"
            stars.append(star_rating)
    stars_df=pd.DataFrame({'stars':stars})


    type_of_traveller=[]
    for _ in range(1,pages+1):
        # Type Of Traveller
        for tr in soup.find_all("tr"):
            td = tr.find("td", class_="review-rating-header type_of_traveller")
            if td and td.text.strip() == "Type Of Traveller":
                type_of_traveller_td = td.find_next_sibling("td", class_="review-value")
                type_of_traveller_text = type_of_traveller_td.text.strip() if type_of_traveller_td else "N/A"
                type_of_traveller.append(type_of_traveller_text)
    traveller_df=pd.DataFrame({'type_of_traveller':type_of_traveller})


    route=[]
    for _ in range(1,pages+1):
        # Route
        for item in soup.find_all('td', class_='review-rating-header route'):
            route_text = item.find_next_sibling('td').text.strip()
            route.append(route_text)
    route_df=pd.DataFrame({'route':route})


    df=pd.concat([df,stars_df,route_df,traveller_df],axis=1)
    df.head()

    df.isna().sum()
    df.dropna(inplace=True)
    df.to_csv(f"{file_name}.csv", index=False)
    
    return prepare(pd.read_csv(f"{file_name}.csv"), f"{file_name}.csv")
