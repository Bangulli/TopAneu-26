import hashlib, os, json
from pathlib import Path

def make_checksums_for_dir(dir, block_size=65536):
    dir = Path(dir)
    for f in [i for i in os.listdir(dir) if i.endswith(".nii.gz")]:
        if (dir/f).is_dir(): continue
        sha256 = hashlib.sha256()
        with open(dir/f, 'rb') as file:
            for block in iter(lambda: file.read(block_size), b''):
                sha256.update(block)
        with open(dir/f.replace(".nii.gz", ".sha"), 'w') as file:
            file.write(sha256.hexdigest())

def check_checksums(checksum_dir, image_dir, block_size=65563):
    checksum_dir = Path(checksum_dir)
    image_dir = Path(image_dir)
    
    for f in [i for i in os.listdir(image_dir) if i.endswith(".nii.gz")]:
        assert (checksum_dir/f.replace(".nii.gz", ".sha")).exists(), f"Couldnt find reference .sha for image {f}"
        sha256 = hashlib.sha256()
        with open(image_dir/f, 'rb') as file:
            for block in iter(lambda: file.read(block_size), b''):
                sha256.update(block)
        with open(checksum_dir/f.replace(".nii.gz", ".sha"), "r") as file:
            ref = file.read()
        assert ref == sha256.hexdigest(), f"Checksums didnt match for image {f}"
    
    print(f"\033[32;1mSUCCESS:\033[0m All checksums matched between checksums in {checksum_dir} and images in {image_dir}")

if __name__ == "__main__":
    make_checksums_for_dir("topaneu_deployment/images")
    check_checksums("topaneu_deployment/images", "topaneu_deployment/images")
    make_checksums_for_dir("topaneu_deployment/location_masks")
    check_checksums("topaneu_deployment/location_masks", "topaneu_deployment/location_masks")
    make_checksums_for_dir("topaneu_deployment/type_masks")
    check_checksums("topaneu_deployment/type_masks", "topaneu_deployment/type_masks")
    make_checksums_for_dir("topaneu_deployment/vessel_masks")
    check_checksums("topaneu_deployment/vessel_masks", "topaneu_deployment/vessel_masks")



