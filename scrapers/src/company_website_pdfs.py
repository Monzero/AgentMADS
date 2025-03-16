import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyPDF2 import PdfReader, PdfWriter
import pandas as pd

def scrape_company_website(companyName):
    """Scrapes policy-related PDFs for a company and merges them into a single PDF."""
    
    # Step 1: Read company website links from Excel
    company_website_df = pd.read_excel("AgentMADS/scrapers/data/links/CompanyWebsiteLInks.xlsx")

    # Step 2: Extract URLs for the given company
    company_row = company_website_df[company_website_df['Company'] == companyName]

    if company_row.empty:
        raise ValueError(f"Company '{companyName}' not found in the dataset.")

    policies_urls = str(company_row.iloc[0]["Investors - Policies"]).strip()
    lodr_regulations = str(company_row.iloc[0]["Inverstors - LoDR"]).strip()

    # Step 3: Create a folder for PDFs
    pdf_folder = f"AgentMADS/scrapers/data/scraped/{companyName}/pdfs"
    os.makedirs(pdf_folder, exist_ok=True)

    def scrape_pdfs_from_url(url):
        """Scrapes all PDF links from the given URL."""
        if not url or url.lower() == "nan":  # Check if the URL is empty or NaN
            return []

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                return [
                    (urljoin(url, link["href"]), link.get_text(strip=True).replace(" ", "_"))
                    for link in soup.find_all("a", href=True)
                    if link["href"].endswith(".pdf")
                ]
        except Exception as e:
            print(f"Error accessing {url}: {e}")
        return []

    # Step 4: Scrape PDFs if URLs are available
    all_pdf_info = scrape_pdfs_from_url(policies_urls) + scrape_pdfs_from_url(lodr_regulations)

    # Step 5: Download each PDF
    downloaded_pdfs = []
    for pdf_url, pdf_name in all_pdf_info:
        pdf_filename = os.path.join(pdf_folder, f"{pdf_name}.pdf")
        try:
            pdf_response = requests.get(pdf_url, timeout=10)
            if pdf_response.status_code == 200:
                with open(pdf_filename, "wb") as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"Downloaded: {pdf_filename}")
                downloaded_pdfs.append(pdf_filename)
            else:
                print(f"Failed to download: {pdf_url}")
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")

    # Step 6: Merge all PDFs into one
    merged_pdf_path = f"AgentMADS/scrapers/data/scraped/{companyName}/combined_policies.pdf"
    pdf_writer = PdfWriter()

    if downloaded_pdfs:
        for pdf_path in downloaded_pdfs:
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    pdf_writer.add_page(page)
            except Exception as e:
                print(f"Error merging {pdf_path}: {e}")
    else:
        # Create a blank PDF if no PDFs were downloaded
        pdf_writer.add_blank_page(width=612, height=792)

    # Step 7: Save the merged PDF
    with open(merged_pdf_path, "wb") as merged_pdf:
        pdf_writer.write(merged_pdf)

    print(f"Combined PDF saved as: {merged_pdf_path}")
    return merged_pdf_path
