import requests, zipfile, tqdm, os
from checksums import check_checksums


url = "https://drive.switch.ch/index.php/s/O36U43RkChkNcHd/download"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open("TopAneu-26.zip", "wb") as f:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192), desc="Receiving Chunks"):
            f.write(chunk)
            
with zipfile.ZipFile("TopAneu-26.zip") as zip:
    zip.extractall("tmp")
    
os.rename("tmp/topaneu_deployment", "TopAneu-26")
os.rename("TopAneu-26.zip", "TopAneu-26/TopAneu-26.zip")
os.rmdir("tmp")
    
for dir in ["images", "location_masks", "type_masks", "vessel_masks"]:
    check_checksums(f"topaneu_deployment/{dir}", f"TopAneu-26/{dir}")