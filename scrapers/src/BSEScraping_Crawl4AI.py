import asyncio
import pandas as pd
from crawl4ai import AsyncWebCrawler

async def bse_scrape_data(companyName):
    # Read URLs and headers from generated_urls.xlsx
    input_file = f"scrapers/data/scraped/{companyName}/generated_urls.xlsx"
    output_file = f"scrapers/data/scraped/{companyName}/scraped_results.xlsx"

    # Load the Excel file into a DataFrame
    try:
        df = pd.read_excel(input_file)
        if "Header" not in df.columns or "URL" not in df.columns:
            raise ValueError("The input Excel file must have 'Header' and 'URL' columns.")
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    # Initialize an empty list to store scraped results
    scraped_data = []

    async with AsyncWebCrawler() as crawler:
        for _, row in df.iterrows():
            header = row["Header"]
            url = row["URL"]
            try:
                print(f"Scraping URL: {url}")
                result = await crawler.arun(url=url)  # Scrape the data
                scraped_data.append({"Header": header, "URL": url, "Scraped Data": result.markdown})  # Store header and result
            except Exception as e:
                print(f"Error processing {url}: {e}")
                scraped_data.append({"Header": header, "URL": url, "Scraped Data": f"Error: {e}"})  # Save error info

    # Save the results to an Excel file
    try:
        output_df = pd.DataFrame(scraped_data)
        output_df.to_excel(output_file, index=False)
        print(f"Scraped data saved to {output_file}")
    except Exception as e:
        print(f"Error saving the results to an Excel file: {e}")
