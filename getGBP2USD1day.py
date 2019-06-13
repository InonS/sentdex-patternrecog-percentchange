from StringIO import StringIO
from zipfile import ZipFile

from requests import get

GBPUSD_SRC_URL = "https://pythonprogramming.net/static/downloads/forex-hft-pattern-recognition/GBPUSD.zip"
GBPUSD_TGT_DIR = "GBPUSD"


def main():
    response = get(GBPUSD_SRC_URL, stream=True)
    if response.ok:
        response_content_as_str = StringIO(response.content)
        response_content_as_zip_file = ZipFile(response_content_as_str)
        response_content_as_zip_file.extractall(path=GBPUSD_TGT_DIR)
