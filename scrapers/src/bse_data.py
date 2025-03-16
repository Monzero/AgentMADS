import os
import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def initialize_driver():
    """Initialize and return the Selenium WebDriver."""
    options = Options()
    options.add_argument("--start-maximized")  # Open browser in full-screen
    driver = webdriver.Chrome(options=options)
    return driver


def wait_for_element(driver, url, info_type):
    """Open the URL and wait until the target element is displayed."""
    driver.get(url)
    if info_type.lower() == 'related party transactions':
        xpath='//*[@id="deribody"]//table[4]'
    else:
        xpath='//*[@id="deribody"]'

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
        element = driver.find_element(By.XPATH, xpath)
        print("Element is displayed:", element.is_displayed())
        return element
    except Exception as e:
        print(f"Error: {e}")
        driver.quit()
        return None


def capture_screenshots(driver, element, output_dir):
    """Scroll through the element and take screenshots."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Get element height
    element_height_scroll = driver.execute_script("return arguments[0].scrollHeight;", element)
    element_height_offset = driver.execute_script("return arguments[0].offsetHeight;", element)
    total_height = max(element_height_scroll, element_height_offset)
    viewport_height = driver.execute_script("return window.innerHeight;")

    print(f"Total Element Height: {total_height}, Viewport Height: {viewport_height}")

    # Scroll and take screenshots
    scroll_position = 0
    screenshot_index = 1
    screenshot_files = []

    while scroll_position < total_height:
        screenshot_name = os.path.join(output_dir, f"screenshot_{screenshot_index}.png")
        driver.save_screenshot(screenshot_name)
        screenshot_files.append(screenshot_name)
        print(f"Saved: {screenshot_name}")

        driver.execute_script(f"window.scrollBy(0, {viewport_height});")
        time.sleep(1)  # Allow time for page load

        scroll_position += viewport_height
        screenshot_index += 1

    print("Scrolling and screenshots completed!")
    return screenshot_files


def merge_screenshots(screenshot_files, info_type, output_dir):
    output_file = f"{output_dir}/{info_type}.png"
    """Merge multiple screenshots into one image."""
    if not screenshot_files:
        print("No screenshots found!")
        return

    images = [Image.open(img) for img in screenshot_files]
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    merged_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height

    merged_image.save(output_file)
    print(f"Merged screenshot saved as '{output_file}'")
    return output_file


def cleanup_screenshots(screenshot_files, keep_file):
    """Delete all screenshots except the merged one."""
    for file in screenshot_files:
        if file != keep_file:
            os.remove(file)
            print(f"Deleted: {file}")       


# Main execution
def get_screenshot(url, info_type, company_name):
    url = url
    output_dir = f"AgentMADS/scrapers/data/scraped/{company_name}"

    driver = initialize_driver()
    element = wait_for_element(driver, url, info_type)

    if element:
        screenshot_files = capture_screenshots(driver, element, output_dir)
        merged_image = merge_screenshots(screenshot_files, info_type, output_dir)
        cleanup_screenshots(screenshot_files, merged_image)

    driver.quit()
    path = os.path.join(output_dir, f"{info_type}.png")
    print(path)
    return os.path.join(output_dir, f"{info_type}.png")
