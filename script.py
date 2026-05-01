from pathlib import Path
import os

path = Path("./faces/person")

images = os.listdir(path)

images.sort()

for i, filename in enumerate(images, start=1):
    
    
    new_name = f"img{i}.png"
    
    old_path = os.path.join(path, filename)
    new_path = os.path.join(path, new_name)
    
    os.rename(old_path, new_path)

print("Done renaming!")