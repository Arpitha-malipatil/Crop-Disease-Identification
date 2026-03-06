import os
import subprocess
import gdown

# ==========================================
# 1. DEFINE THE DATA MANAGER CLASS
# ==========================================
class CropDataManager:
    """A class to handle downloading and organizing crop datasets."""
    
    def __init__(self, base_folder="./my_crop_data"):
        """Initializes the manager and creates a base folder for our data."""
        self.base_folder = base_folder
        os.makedirs(self.base_folder, exist_ok=True)
        print(f"Data manager initialized. Files will be saved to: {self.base_folder}")

    def download_kaggle_plantvillage(self):
        """Downloads and unzips the PlantVillage dataset using Kaggle's API."""
        print("\n--- Starting Kaggle dataset download ---")
        dataset_name = "abdallahalidev/plantvillage-dataset"
        save_path = os.path.join(self.base_folder, "plantvillage")
        
        # We use subprocess to run the Kaggle CLI command directly
        command = [
            "kaggle", "datasets", "download", 
            "-d", dataset_name, 
            "-p", save_path, 
            "--unzip"
        ]
        
        # Execute the command in the terminal
        subprocess.run(command)
        print(f"Kaggle download complete! Saved in {save_path}")

    def download_drive_folder(self, drive_url):
        """Downloads a public folder from Google Drive using gdown."""
        print("\n--- Starting Google Drive download ---")
        save_path = os.path.join(self.base_folder, "drive_data")
        
        # gdown allows us to download whole folders using the URL
        gdown.download_folder(drive_url, output=save_path, quiet=False, use_cookies=False)
        print(f"Drive download complete! Saved in {save_path}")

# ==========================================
# 2. EXECUTE THE CODE
# ==========================================
def main():
    # Step A: Create an instance of our class
    data_loader = CropDataManager(base_folder="./my_crop_data")

    # Step B: Trigger the Kaggle download
    data_loader.download_kaggle_plantvillage()

    # Step C: Trigger the Google Drive download
    drive_link = "https://drive.google.com/drive/folders/1aY7aQl5uCr3LrO8Wgni3oJvbHFp991AO?usp=drive_link"
    data_loader.download_drive_folder(drive_link)

    print("\nAll datasets are successfully downloaded and ready for the next step!")

# This ensures the main() function only runs if we execute this file directly
if __name__ == "__main__":
    main()