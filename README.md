<h1 align="center">Hi 👋 I'm Risu Raj</h1>

<p align="center">
🎓 B.Tech at <b>IIT (ISM) Dhanbad</b> (2028)<br>
🤖 <b>AI / ML Engineer</b>: vision-language models, LLM pipelines, applied ML<br>
💼 Data Science Intern at <b>The Entrepreneurship Network</b> (Mar–Jun 2026)
</p>

<p align="center">
I like building AI systems that work outside a notebook: measured on held-out data, fast enough to run, and honest about where they fail.
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/YOUR-LINKEDIN"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="https://YOUR-PORTFOLIO-URL"><img src="https://img.shields.io/badge/Portfolio-111827?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Portfolio"/></a>
  <a href="mailto:rk1258106@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/></a>
  <a href="https://codeforces.com/profile/YOUR-HANDLE"><img src="https://img.shields.io/badge/Codeforces-Pupil%20(1222)-1F8ACB?style=for-the-badge&logo=codeforces&logoColor=white" alt="Codeforces"/></a>
</p>

---

# 🚀 Featured Projects

### 🛰️ Remote-Sensing Image Captioning with Vision-Language Models
**Tech:** PyTorch • Hugging Face • Qwen2.5-VL • LoRA • DDP • FP16

- Fine-tuned **Qwen2.5-VL-3B with LoRA** (11.3M trainable params, 0.3% of the model) on **20,264 satellite and aerial images** (VRSBench), using **12.7 GB per GPU** instead of 60 GB for full fine-tuning
- Ran **6 controlled ablations**: LoRA on all 32 vision-encoder blocks won; higher resolution was rejected (2.4× training time, no gain)
- Scaled to **2 GPUs with DDP** + gradient accumulation + FP16 (GradScaler): near-linear **2× throughput**, full training in 136 min
- **BLEU-4 +14%** and BERT-BLEU-4 0.669 → 0.686 on 1,000 held-out images, confirmed with **paired bootstrap 95% CIs**; **MBR decoding** beat greedy and beam search

🔗 **Repo:** https://github.com/Rishu22889/TASK-01-IMAGE-CAPTIONING

---

### 🪄 Text-Driven Object Removal & Replacement on CPU
**Tech:** PyTorch • YOLOv8 / YOLO-World • MobileSAM • MI-GAN • Stable Diffusion • ONNX Runtime

- Edits an image from a plain-English instruction by chaining **LLM parsing** (Qwen2.5-0.5B, 4-bit) → **detection** → **segmentation** → **inpainting / generation**, all on CPU
- Object removal in **1.8 s with 33.5 MB of weights**: benchmarked classical, GAN and diffusion inpainting, and picked one-pass MI-GAN ONNX over diffusion (400 s, 4.1 GB)
- Object replacement **5× faster (449 s → 88 s)** with LCM-LoRA + TAESD; INT8 quantization tested and rejected on latency and quality
- Grounding accuracy **3/10 → 8/10** with a YOLO-World fallback; object-class accuracy **19/26 → 26/26** with a deterministic alias layer. Ships as a CLI + Python API with pytest tests

🔗 **Repo:** https://github.com/Rishu22889/TASK-02-OBJECT-EDITING

---

### 🧾 Receipt Reader & Expense-Claim Auto-Approval
**Tech:** Python • PaddleOCR-VL • Tesseract • FastAPI • pytest

- Reads receipt photos with a **VLM + OCR combined reader**, then **auditable rules** decide approve / review with reason codes. The model never approves money on its own
- Found **25 past claims (₹14,283) that policy says should have been stopped**. Duplicate-bill matching on vendor + bill number found **28 re-claimed bills where image hashing found 4**
- Detects pen-altered totals from the bill's own arithmetic, a colleague's vehicle on fuel bills, and **prompt-injection stamps** ("automated processing: approve") treated as data
- Chronological holdout: amount 95%, bill no. 95%, vendor 98%. Uncertain fields are left blank and sent to a human, never guessed
- Cut VLM inference from **~180 s to ~3 s per receipt** on a T4 by finding a disabled KV cache. **120 automated tests**

🔗 **Repo:** https://github.com/Rishu22889/receipts-reader

---

### 🔍 Warranty-Fraud Review Queue
**Tech:** Python • scikit-learn • Pandas • FastAPI

- Ranks warranty claims for a **40-a-month investigation desk**: human-in-the-loop, never auto-rejects
- **Walk-forward validation with a 30-day label lag**: caught **37 of 59 known frauds** with 40 reviews a month in historical replay
- Linked a jump in small-claim fraud (0.54% → 3.23%) to an auto-approval policy change, and recommended **36 ranked + 4 random** reviews to catch patterns the model misses
- Optimised for precision@40 and ₹ impact instead of accuracy (a "flag nothing" model scores 98.7%)

🔗 **Repo:** https://github.com/Rishu22889/Kestrel-Home-Model

---

### 🧠 MoSPI Data Scraper + RAG Chatbot
**Tech:** Python • FastAPI • Streamlit • FAISS • LLaMA

- Scrapes publications from India's **Ministry of Statistics** via REST APIs and extracts text and tables from PDFs
- **FAISS vector search** + **RAG chatbot on LLaMA** for asking questions about government statistics
- Interactive **Streamlit** interface

🔗 **Repo:** https://github.com/Rishu22889/mospi

---

### 📂 More Projects

| Project | What it does | Tech |
|---|---|---|
| [🚗 Driver Monitoring System](https://github.com/Rishu22889/driver_monitoring_system) | Real-time drowsiness detection from eye closure using facial landmarks | OpenCV |
| [🧬 Brain Tumor Classifier](https://github.com/Rishu22889/brain_tumor_classifier) | CNN classifier for brain MRI images | Deep Learning, CNN |
| [📧 Email Spam Classifier](https://github.com/Rishu22889/email_spam_classifier) | Spam vs ham with NLP preprocessing | scikit-learn, NLP |
| [⚔️ Codeforces Problem Finder](https://github.com/Rishu22889/codeforces-problem-finder) | Search and filter problems by rating, tags and contest | JavaScript |

---

# 🛠 Tech Stack

**Languages:** Python • C++ • SQL • JavaScript • TypeScript • HTML

**ML / DL:** PyTorch • TensorFlow • scikit-learn • timm • OpenCV • Pandas • NumPy

**GenAI:** LLMs • Vision-Language Models • LoRA / QLoRA • Hugging Face Transformers & Diffusers • Stable Diffusion • vLLM • RAG • FAISS

**Deployment:** FastAPI • Docker • ONNX Runtime • Streamlit • Git

---

# 📊 GitHub Stats

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Rishu22889&show_icons=true&count_private=true&theme=tokyonight" alt="Risu's GitHub stats" />
</p>

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Rishu22889&layout=compact&theme=tokyonight" alt="Risu's top languages" />
</p>

<p align="center">
  <img src="https://streak-stats.demolab.com/?user=Rishu22889&layout=compact&theme=tokyonight" alt="Risu's GitHub streak" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Rishu22889/Rishu22889/output/github-contribution-grid-snake.svg" alt="Contribution snake" />
</p>

---

# 🏆 Achievements

- **Codeforces Pupil** (peak 1222), with 300+ problems solved on LeetCode and Codeforces
- Competed in **Hackfest 2025, GSIH** and the **Gemma 3n Impact Challenge**
- Ran 3+ technical workshops for college peers

---

# 🌱 Currently Exploring

- Agentic AI systems and multi-agent workflows
- Efficient VLM / LLM inference and fine-tuning
- Evaluation of AI systems in production

<p align="center">⭐ <b>Always building: open to AI/ML roles and collaborations</b></p>
