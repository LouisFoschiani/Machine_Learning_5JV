import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def is_valid_image(file_url):
    # Vérifie si l'URL de l'image se termine par .png, .jpg ou .jpeg
    return file_url.lower().endswith(('.png', '.jpg', '.jpeg'))

def download_image(image_url, folder_path):
    # Vérifie si l'URL de l'image est valide
    if not is_valid_image(image_url):
        print(f"Ignoré (format non supporté) : {image_url}")
        return

    # Vérifie si le dossier existe, sinon le crée
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    response = requests.get(image_url)
    if response.status_code == 200:
        image_filename = os.path.join(folder_path, image_url.split("/")[-1])

        try:
            with open(image_filename, 'wb') as file:
                file.write(response.content)
            print(f"Image enregistrée : {image_filename}")
        except IOError as e:
            print(f"Erreur lors de l'enregistrement de l'image : {e}")

def scrape_images(url, folder_path='C:\\Users\\Louis\\Documents\\GitHub\\machine_Learning_5JV\\images\\scraping\\orange'):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for img_tag in soup.find_all('img'):
        img_url = img_tag.get('src')
        img_url = urljoin(url, img_url)
        download_image(img_url, folder_path)

if __name__ == "__main__":
    for i in range(1, 101):  # Pour les pages de 1 à 100
        page_url = f"https://stock.adobe.com/fr/search?filters%5Bcontent_type%3Aphoto%5D=1&filters%5Bcontent_type%3Aimage%5D=1&filters%5Breleases%3Ais_exclude%5D=1&filters%5Borientation%5D=square&k=orange+fruit&order=relevance&safe_search=1&limit=100&search_page={i}&load_type=page&search_type=filter-select&acp=&aco=orange+fruit&scoring%5Bae_depth_of_field%5D=0&get_facets=1"
        print(f"Scraping de la page : {page_url}")
        scrape_images(page_url)
