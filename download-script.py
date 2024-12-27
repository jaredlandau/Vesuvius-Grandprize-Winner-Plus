import os
import re
import requests
from requests.auth import HTTPBasicAuth

# Configuration
base_url_template = "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths/{id}/"
output_folder = "layers"
username = input("Enter username: ")
password = input("Enter password: ")

# Prompt user for ID and file range
scroll_id = input("Enter the scroll ID (e.g., 20241108111522): ").strip()
start_range = int(input("Enter the starting file number: "))
end_range = int(input("Enter the ending file number: "))

# Construct base URL and folder paths
base_url = base_url_template.format(id=scroll_id)
layers_url = base_url + "layers/"
layers_folder = os.path.join("downloaded_images", "layers")
os.makedirs(layers_folder, exist_ok=True)

# Download .jpg files into layers folder
for i in range(start_range, end_range + 1):
    file_name = f"{i}.jpg"
    url = f"{layers_url}{file_name}"
    output_path = os.path.join(layers_folder, file_name)

    try:
        print(f"Downloading {file_name} from layers directory...")
        response = requests.get(url, auth=HTTPBasicAuth(username, password), stream=True)

        if response.status_code == 200:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Saved {file_name} to {layers_folder}")
        else:
            print(f"Failed to download {file_name}: {response.status_code} {response.reason}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")

# Go back to the home directory to find _mask.png files
print("\nChecking for _mask.png files in the home directory...")
try:
    response = requests.get(base_url, auth=HTTPBasicAuth(username, password))
    if response.status_code == 200:
        # Find all _mask.png files using regex
        mask_files = re.findall(r'href="([^"]*_mask\.png)"', response.text)
        if mask_files:
            for file_name in mask_files:
                url = f"{base_url}{file_name}"
                output_path = os.path.join(os.getcwd(), file_name)

                try:
                    print(f"Downloading {file_name}...")
                    file_response = requests.get(url, auth=HTTPBasicAuth(username, password), stream=True)

                    if file_response.status_code == 200:
                        with open(output_path, "wb") as file:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        print(f"Saved {file_name} to {os.getcwd()}")
                    else:
                        print(f"Failed to download {file_name}: {file_response.status_code} {file_response.reason}")
                except Exception as e:
                    print(f"Error downloading {file_name}: {e}")
        else:
            print("No _mask.png files found in the home directory.")
    else:
        print(f"Failed to access the home directory: {response.status_code} {response.reason}")
except Exception as e:
    print(f"Error accessing the home directory: {e}")

print("\nDownload complete!")