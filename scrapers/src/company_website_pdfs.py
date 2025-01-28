import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyPDF2 import PdfReader
import pandas as pd

def scrape_company_website(companyName):
    # Step 1: Define the URL of the page
    company_website_df = pd.read_excel("AgentMADS/scrapers/data/links/CompanyWebsiteLInks.xlsx")

    # Iterate over each company in the company data
    company_found = False
    for _, company_row in company_website_df.iterrows():
        if company_row['Company'] == companyName:
            # Extract company URLs
            corporate_governance_report_url = str(company_row["Corporate Governance Company website"])
            policies_urls = str(company_row["Investors - Policies"])
            company_found = True
            break

    if not company_found:
        raise ValueError(f"Company '{companyName}' not found in the dataset.")

    # Create folders for PDFs and extracted text
    pdf_folder = f"AgentMADS/scrapers/data/scraped/{companyName}/pdfs"
    os.makedirs(pdf_folder, exist_ok=True)

    # Step 3: Function to scrape PDFs from a given URL
    def scrape_pdfs_from_url(url, folder):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
        
            # Step 4: Find all PDF links and their names
            pdf_info = []  # Store tuples of (PDF URL, Name)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.endswith(".pdf"):
                    pdf_url = urljoin(url, href)  # Resolve relative URLs to absolute
                    pdf_name = link.get_text(strip=True).replace(" ", "_")  # Extract name from the link text
                    pdf_info.append((pdf_url, pdf_name))
            
            return pdf_info
        else:
            print(f"Failed to fetch webpage: {url}")
            return []

    # Step 5: Scrape PDFs from corporate governance report and policies URLs
    all_pdf_info = scrape_pdfs_from_url(corporate_governance_report_url, pdf_folder) + scrape_pdfs_from_url(policies_urls, pdf_folder)

    # Step 6: Download each PDF and extract text
    combined_text = ""
    for pdf_url, pdf_name in all_pdf_info:
        pdf_filename = os.path.join(pdf_folder, f"{pdf_name}.pdf")
        try:
            # Download PDF
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                with open(pdf_filename, "wb") as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"Downloaded: {pdf_filename}")
                
                # Extract text from the PDF
                reader = PdfReader(pdf_filename)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text()
                
                # Append to combined text
                combined_text += f"\n\n=== {pdf_name} ===\n\n" + pdf_text
            else:
                print(f"Failed to download: {pdf_url}")
        except Exception as e:
            print(f"Error processing {pdf_url}: {e}")
    
    # Step 7: Save combined text into a single document
    combined_file = f"AgentMADS/scrapers/data/scraped/{companyName}/combined_policies.txt"
    with open(combined_file, "w", encoding="utf-8") as file:
        file.write(combined_text)
    print(f"All text combined into: {combined_file}")
