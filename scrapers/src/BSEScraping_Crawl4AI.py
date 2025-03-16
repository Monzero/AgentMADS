import os
import pandas as pd
from crawl4ai import AsyncWebCrawler
from lxml import html
from bse_data import get_screenshot
from fpdf import FPDF
from PIL import Image
import re

async def bse_scrape_data(company_name):
    """Scrape data from BSE and save to Excel & PDF."""
    
    input_file = f"AgentMADS/scrapers/data/scraped/{company_name}/bse_generated_urls.xlsx"
    output_file = f"AgentMADS/scrapers/data/scraped/{company_name}/bse_scraped_results.xlsx"

    # Ensure input/output files exist
    for file in (input_file, output_file):
        create_empty_excel_if_not_exists(file)

    # Load Excel
    try:
        df = pd.read_excel(input_file)
        required_cols = {"Section", "Header", "URL"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Input Excel must have 'Section', 'Header', and 'URL' columns.")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    scraped_data = []
    async with AsyncWebCrawler() as crawler:
        for _, row in df.iterrows():
            section, header, url = row["Section"], row["Header"], row["URL"]
            print(f"Scraping: {url}")

            try:
                # Handle screenshots for specific headers
                if header.lower() in {"corporate governance", "shareholding pattern", "related party transactions"}:
                    result = get_screenshot(url, header, company_name)
                else:
                    result = await crawler.arun(url=url)

                scraped_data.append({
                    "Section": section,
                    "Header": header,
                    "URL": url,
                    "Scraped Data": result if isinstance(result, str) else result.markdown
                })

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                scraped_data.append({"Section": section, "Header": header, "URL": url, "Scraped Data": f"Error: {e}"})

    # Save scraped data to Excel
    try:
        pd.DataFrame(scraped_data).to_excel(output_file, index=False)
        print(f"Scraped data saved: {output_file}")
    except Exception as e:
        print(f"Error saving Excel: {e}")

    # Convert results to PDFs
    try:
        save_to_pdf(output_file)
    except Exception as e:
        print(f"Error generating PDF: {e}")


def create_empty_excel_if_not_exists(file_path):
    """Ensure an Excel file exists, create an empty one if not."""
    if not os.path.exists(file_path):
        pd.DataFrame().to_excel(file_path, index=False, engine="openpyxl")



def save_to_pdf(excel_file):
    """Converts scraped data to a single PDF grouped by 'Section' with headers as subheaders."""
    df = pd.read_excel(excel_file)

    # Create PDF document
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Iterate over sections
    for section, group in df.groupby("Section"):
        pdf.add_page()
        
        # Add Section Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, f"Section: {section}", ln=True, align="C")
        pdf.ln(8)

        for _, row in group.iterrows():
            header = row["Header"]
            scraped_data = str(row["Scraped Data"]).strip()

            # Add Header as Subheader
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 8, f"--- {header} ---", ln=True)
            pdf.ln(3)

            # Add Text Data
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 8, clean_text(scraped_data))
            pdf.ln(5)

            # Add Image if it's a valid path
            if is_valid_image(scraped_data):
                add_image_to_pdf(pdf, scraped_data)

    # Save the single combined PDF
    file_path = os.path.dirname(excel_file)
    pdf_filename = os.path.join(file_path, "cg_report.pdf")
    pdf.output(pdf_filename)
    print(f"PDF saved: {pdf_filename}")


def is_valid_image(file_path):
    """Check if the file exists and is an image."""
    return isinstance(file_path, str) and file_path.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(file_path)


def add_image_to_pdf(pdf, image_path):
    """Add an image to a new PDF page, ensuring it fills the page without overlapping text."""
    try:
        image = Image.open(image_path)
        page_width, page_height = 210, 297  # A4 size in mm

        # Get original image size
        width, height = image.size

        # Scale to fit full page while maintaining aspect ratio
        scale = min(page_width / width, page_height / height)
        new_width, new_height = int(width * scale), int(height * scale)

        # Center image
        x_offset = (page_width - new_width) / 2
        y_offset = (page_height - new_height) / 2

        # **Add a new page and insert only the image**
        pdf.add_page()
        pdf.image(image_path, x=x_offset, y=y_offset, w=new_width, h=new_height)

        # **Force a new page after the image to prevent text overlap**
        pdf.add_page()

    except Exception as e:
        pdf.multi_cell(0, 8, f"[Error displaying image: {e}]")


def clean_text(text):
    """Remove unsupported characters and replace fancy quotes."""
    if not isinstance(text, str):
        return text  # Return unchanged if not a string
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters

