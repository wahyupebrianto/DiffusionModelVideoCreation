import os
import argparse

def rename_frames(folder):
    folder = os.path.abspath(folder)
    files = sorted([f for f in os.listdir(folder) if f.startswith("frame_") and f.endswith(".png")])

    # Tahap 1: Rename ke nama sementara
    for i, fname in enumerate(files):
        src = os.path.join(folder, fname)
        temp_name = f"temp_{i:05d}.png"
        dst = os.path.join(folder, temp_name)
        os.rename(src, dst)

    # Tahap 2: Rename ke format final
    temp_files = sorted([f for f in os.listdir(folder) if f.startswith("temp_") and f.endswith(".png")])
    for i, fname in enumerate(temp_files):
        src = os.path.join(folder, fname)
        final_name = f"frame_{i+1:05d}.png"
        dst = os.path.join(folder, final_name)
        os.rename(src, dst)
        print(f"{fname} -> {final_name}")

    print(f"\n✅ Semua file dalam '{folder}' berhasil diurutkan ulang!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename frame_XXXXX.png files to continuous numbering.")
    parser.add_argument("folder", help="Path ke folder berisi frame gambar")
    args = parser.parse_args()

    rename_frames(args.folder)
