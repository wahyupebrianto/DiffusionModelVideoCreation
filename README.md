# 🎨 AI Dream Video Generator

Proyek ini menghasilkan video bergaya mimpi (dreamlike) dari teks menggunakan model pretrained *Stable Diffusion*, lalu menggabungkannya menjadi video dan menambahkan musik latar.

---

## 🔍 Tentang Model

Proyek ini menggunakan [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), sebuah model difusi teks-ke-gambar buatan **CompVis**, yang memungkinkan Anda menghasilkan gambar resolusi tinggi hanya dari teks deskriptif.

* **Arsitektur**: Latent Diffusion Model (LDM)
* **Tokenizer**: CLIP tokenizer dari `openai/clip-vit-large-patch14`
* **Ukuran model**: \~4GB
* **Kebutuhan sistem**: GPU sangat disarankan

---

## 📋 Cara Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/username/namarepo.git
cd namarepo
```

### 2. (Opsional) Siapkan virtual environment

```bash
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan skrip untuk menghasilkan gambar

```bash
python dreaming.py --prompt "sunset over a futuristic ocean, glowing orbs flying, surreal and peaceful atmosphere" --output_dir outputs/forestdream
```

### 5. Konversi gambar menjadi video menggunakan FFmpeg

```bash
ffmpeg -framerate 10 -i outputs/forestdream/frame%05d.jpg -c:v libx264 -pix_fmt yuv420p video.mp4
```

### 6. Tambahkan musik latar

```bash
ffmpeg -i video.mp4 -i music.mp3 -shortest -c:v copy -c:a aac final_with_music.mp4
```

### 7. Upload manual ke YouTube

---

## 🧠 Parameter `dreaming.py`

```bash
--prompt              : Deskripsi teks untuk menghasilkan gambar
--num_inference_steps : Jumlah langkah difusi (default: 50)
--guidance_scale      : Tingkat "kesesuaian" antara teks dan gambar (default: 7.5)
--output_dir          : Folder penyimpanan hasil
--num_frames          : Jumlah frame gambar yang ingin dibuat
```

---

## 💾 Contoh Output

* Folder `outputs/forestdream/` berisi gambar `frame00000.jpg`, `frame00001.jpg`, dst.
* `video.mp4` adalah hasil dari penggabungan gambar
* `final_with_music.mp4` adalah video akhir dengan musik

---

## 📦 Ketergantungan Utama

* `diffusers`
* `transformers`
* `torch`
* `fire`
* `Pillow`
* `numpy`
* `ffmpeg` (pastikan tersedia di PATH sistem)

---

## 📜 Referensi

* [Stable Diffusion v1-4 - CompVis (HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v1-4)
* [Diffusers Library by Hugging Face](https://github.com/huggingface/diffusers)
* [FFmpeg Documentation](https://ffmpeg.org/documentation.html)





0:57-7:46

- python dreaming.py --prompt "a swirling galaxy of vibrant stars, nebulae, and cosmic clouds, with ethereal colors blending into an infinite spiral" --output_dir "outputs" --num_frames 1800 --steps_per_transition 20 --num_inference_steps 50 --guidance_scale 7.5 --width 512 --height 512 --seed 42 --model_id "CompVis/stable-diffusion-v1-4" --scheduler_type "LMS" --device "cuda"
- python dreaming.py --prompt "The silhouette of a man standing tall beside a stack of books, balanced over a crystal-clear ocean, gazing in awe at planet Earth before him, surrounded by infinite stars and the vast cosmos, cinematic, surreal, high detail, cosmic lighting." --output_dir "outputs" --num_frames 1800 --steps_per_transition 20 --num_inference_steps 50 --guidance_scale 7.5 --width 512 --height 512 --seed 42 --model_id "CompVis/stable-diffusion-v1-4" --scheduler_type "LMS" --device "cuda"


- video
ffmpeg -framerate 8 -i swirling_galaxyframe_%05d.png -c:v libx264 -pix_fmt yuv420p video.mp4


- Prompt
a dreamy vision of an astronaut adrift in the cosmos, mesmerized by the glowing Earth below and the infinite stars above, ethereal, surreal, cosmic art, nebula colors, peaceful and profound


bayangan seseorang pria berdiri tegak di samping tumpukan Buku berpijak pada samudra bening, menghadap kosmos terpesona oleh Bumi yang bersinar di depan dan bintang-bintang tak terbatas di sekitar.