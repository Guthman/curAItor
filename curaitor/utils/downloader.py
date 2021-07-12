import requests
from tqdm.auto import tqdm
from pathlib import Path


# Utility function to download files
def download_file(url, download_location):
    local_path = Path(download_location) / url.split('/')[-1]
    if not local_path.is_file():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                pbar = tqdm(total=int(r.headers['Content-Length']))
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    return local_path
