# 🐳 Docker Deployment & CLI Usage

> Ensuring seamless evaluation and reproducibility, the **Code Red AI** tracker is packaged using Docker. This allows judges to run inference on new videos via a simple Command-Line Interface (CLI) — no manual Python environment setup, dependency management, or path configuration required.

---

## 📦 Building the Docker Image

Navigate to the **root directory** of the project (where the `Dockerfile` is located) and run:

```bash
docker build -t aic-tracker .
```

---

## ▶️ Running Inference via CLI

To process a video and generate tracking results, use the `docker run` command below.  
You must **mount a local directory** containing the test video and its initial annotation file to the container using the Docker volume flag (`-v`).

```bash
docker run --rm \
  -v /absolute/path/to/local/data:/data \
  aic-tracker \
  --video /data/test_video.mp4 \
  --annot /data/annotation.txt \
  --output /data/final_submission.csv
```

---

## 🔧 Command Arguments Explained

| Argument | Description |
|---|---|
| `--rm` | Automatically removes the container after execution finishes — keeps the system clean and frees up resources. |
| `-v /local/path:/data` | Mounts your local folder (containing the video and annotation files) to a virtual `/data` directory inside the Docker container. |
| `--video` | Path to the input video file **as seen from inside the container** (e.g., `/data/test_video.mp4`). |
| `--annot` | Path to the initial bounding box annotation file (e.g., `/data/init_bbox.txt`). |
| `--output` | Desired save path for the generated tracking results. The CSV will be saved directly to your local mounted directory. |

---

## 📁 Example Directory Structure

```
/absolute/path/to/local/data/
├── test_video.mp4
├── annotation.txt
└── final_submission.csv   ← generated after running
```

---

## ✅ Quick Checklist

- [ ] Docker is installed and running
- [ ] Docker image is built (`docker build -t aic-tracker .`)
- [ ] Local data directory contains the video and annotation files
- [ ] Absolute path is used in the `-v` volume mount flag
