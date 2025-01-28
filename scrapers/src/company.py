import pandas as pd
import os

# Prepare a list to store the generated URLs
generated_urls = []
Qname = 'December 2024'
qtrid = 124

def get_company_details(companyName):
    """
    Fetch company details for a given company name from the companies.xlsx file.
    """
    # Load the Excel file containing company data
    company_df = pd.read_excel("AgentMADS/scrapers/data/links/Companies.xlsx")

    # Iterate over each company in the company data
    for _, company_row in company_df.iterrows():
        if company_row['Company Name'] == companyName:
            # Extract company details
            company_details = {
                "Company Name": str(company_row["Company Name"]),
                "scriptcd": str(company_row["scriptcd"]),
                "scode": str(company_row["scode"]),
                "comName": company_row["comName"].replace(" ", "%20"),  # Encode spaces in URLs
                "Qname": Qname.replace(" ", "%20"),      # Encode spaces in URLs
                "qtrid": qtrid,
                "sname1": company_row["sname1"],
                "sname2": company_row["sname2"]
            }
            return company_details  # Return details when the company is found

    raise ValueError(f"Company '{companyName}' not found in the dataset.")

def get_generated_urls(companyDetails):
    """
    Generate URLs based on company details and templates from BSELinks.xlsx.
    """
    # Load the Excel file containing URL templates
    links_df = pd.read_excel("AgentMADS/scrapers/data/links/BSELinks.xlsx")
    
    # Iterate over each link template
    for _, link_row in links_df.iterrows():
        header = link_row["Header"]
        template_url = link_row["Links"]
        
        # Ensure template_url is a string
        if isinstance(template_url, str):
            try:
                # Format the URL with proper values
                formatted_url = template_url.format(
                    **{
                        k: int(v) if isinstance(v, float) else v
                        for k, v in companyDetails.items()
                    }
                )
                generated_urls.append({"Header": header, "Company Name": companyDetails["Company Name"], "URL": formatted_url})
            except KeyError as e:
                print(f"Missing placeholder in template for Header: {header}, Error: {e}")
        else:
            print(f"Skipping invalid template URL for Header: {header}")

# Function to save generated URLs to an Excel file
def save_generated_urls(companyName):
    # check for company folder
    create_company_folder(companyName)

    output_file = f"AgentMADS/scrapers/data/scraped/{companyName}/generated_urls.xlsx"
    """
    Save all generated URLs to a new Excel file.
    """
    # Convert the list of generated URLs to a DataFrame
    output_df = pd.DataFrame(generated_urls)
    output_df.to_excel(output_file, index=False)
    print("URLs generated and saved to 'generated_urls.xlsx'")


def create_company_folder(companyName):
    # Define the path for the directory
    dir_path = f"AgentMADS/scrapers/data/scraped/{companyName}"

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

