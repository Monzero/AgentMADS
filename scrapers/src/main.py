from company import get_company_details, get_generated_urls, save_generated_urls
from company_website_pdfs import scrape_company_website
from BSEScraping_Crawl4AI import bse_scrape_data
import asyncio


# Main logic: Call the functions for a specific company
if __name__ == "__main__":
    try:
        # Specify the company name
        company_name = "Shree Cement"
        
        # Get company details
        company_details = get_company_details(company_name)
        
        # Generate URLs for the company
        get_generated_urls(company_details)
        
        # Save the results to an Excel file
        save_generated_urls(company_name)

        # scrape BSE links and save to .xlsx file
        asyncio.run(bse_scrape_data(company_name))

        # scrape company website
        scrape_company_website(company_name)

    except Exception as e:
        print(f"An error occurred: {e}")