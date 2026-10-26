import os
from pathlib import Path  

skript_root = Path(__file__).resolve().parent
root_dir = Path("RUGD_frames-with-annotations")  
result_dir = skript_root / "RUGD_frames_database"

result_dir.mkdir(exist_ok=True)

png_files = []

#rekurzivne vyhladanie .png suborov
png_files = sorted(list(root_dir.rglob("*.png")))

png_files.sort()

#presunutie do result_dir a premenovanie .png suborov
for i, png_path in enumerate(png_files):
    os.rename(png_path, result_dir / f"img_{i:05d}.png")
    print(f"Renamed {png_path} to {result_dir / f'img_{i:05d}.png'}")