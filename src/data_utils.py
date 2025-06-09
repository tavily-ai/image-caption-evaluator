import os
import tarfile
import requests
import pandas as pd
from typing import List


class DatasetManager:
    """
    Handles downloading, extracting, and loading XTD10 dataset captions and images.

    Attributes:
        REPO_ROOT (str): Absolute path to the root of the repository.
        IMAGE_DIR (str): Absolute path where images are stored after extraction.
        ARCHIVE_PATH (str): Absolute path to the downloaded archive file.
        IMAGES_URL (str): URL to download the XTD10 image archive.
        GITHUB_DATA_PATH (str): Base URL for English and most captions.
        GITHUB_DATA_PATH_DE_FR (str): Base URL for German and French captions.
        GITHUB_DATA_PATH_JP (str): Base URL for Japanese captions.
        SUPPORTED_LANGUAGES (List[str]): List of supported language codes for captions.
    """

    REPO_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    IMAGE_DIR: str = os.path.join(REPO_ROOT, "data", "processed", "images")
    ARCHIVE_PATH: str = os.path.join(REPO_ROOT, "data", "raw", "images.tar.gz")

    IMAGES_URL: str = "https://nllb-data.com/test/xtd10/images.tar.gz"

    GITHUB_DATA_PATH: str = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/"
    GITHUB_DATA_PATH_DE_FR: str = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/MIC/"
    GITHUB_DATA_PATH_JP: str = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/STAIR/"
    SUPPORTED_LANGUAGES: List[str] = ["es", "it", "ko", "pl", "ru", "tr", "zh", "en", "de", "fr", "jp"]

    def download_and_extract_images(self) -> None:
        """
        Downloads the XTD10 image archive if not already present and extracts images.

        Creates directories as needed.

        Prints progress messages about download and extraction status.
        """
        os.makedirs(os.path.dirname(self.ARCHIVE_PATH), exist_ok=True)
        if not os.path.exists(self.ARCHIVE_PATH):
            print("Downloading image archive...")
            response = requests.get(self.IMAGES_URL, stream=True)
            with open(self.ARCHIVE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

        if not os.path.exists(self.IMAGE_DIR):
            print("Extracting images...")
            os.makedirs(self.IMAGE_DIR, exist_ok=True)
            with tarfile.open(self.ARCHIVE_PATH, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(self.IMAGE_DIR))
            print("Extraction complete.")
        else:
            print("Images already extracted.")

    def _get_lines(self, url: str) -> List[str]:
        """
        Helper method to download and split text file content into lines.

        Args:
            url: URL of the text file to download.

        Returns:
            List of strings, each a line from the text file.

        Raises:
            requests.HTTPError: If the HTTP request fails.
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text.splitlines()

    def get_captions_dataframe(self, lang_code: str = "en") -> pd.DataFrame:
        """
        Downloads and returns a DataFrame of image filenames and captions for the specified language.

        Args:
            lang_code: Language code for captions, must be in SUPPORTED_LANGUAGES.

        Returns:
            Pandas DataFrame with columns: 'path' (image filename) and 'caption' (text).

        Raises:
            ValueError: If lang_code is not supported or
                        if the number of images and captions do not match.
        """
        if lang_code not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"{lang_code} not in supported languages: {self.SUPPORTED_LANGUAGES}")

        captions_path = self.GITHUB_DATA_PATH
        if lang_code in ["de", "fr"]:
            captions_path = self.GITHUB_DATA_PATH_DE_FR
        elif lang_code == "jp":
            captions_path = self.GITHUB_DATA_PATH_JP

        image_names = self._get_lines(self.GITHUB_DATA_PATH + "test_image_names.txt")
        captions = self._get_lines(captions_path + f"test_1kcaptions_{lang_code}.txt")

        if len(image_names) != len(captions):
            raise ValueError("Mismatch between number of images and captions")

        return pd.DataFrame({"path": image_names, "caption": captions})
