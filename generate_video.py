import os
import subprocess

def images_to_video(
    image_folder="outputs/forestdream",
    output_video="outputs/forestdream.mp4",
    fps=24,
    ffmpeg_path="C:\Program Files\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"  # bisa diubah ke path lengkap misalnya "C:/ffmpeg/bin/ffmpeg.exe"
):
    # Pastikan folder ada
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Folder '{image_folder}' tidak ditemukan.")

    # Format nama frame: frame_00000.png, frame_00001.png, ...
    cmd = [
        ffmpeg_path,
        "-framerate", str(fps),
        "-i", os.path.join(image_folder, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    print("Running ffmpeg to create video...")
    subprocess.run(cmd, check=True)
    print(f"Video saved to {output_video}")

# Contoh pemanggilan
images_to_video()