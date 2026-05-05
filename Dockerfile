# هنبدأ بنسخة بايثون خفيفة
FROM python:3.10-slim

# نحدد مسار العمل جوه الدوكر
WORKDIR /app

# نسطب شوية ملفات نظام ضرورية لمكتبة OpenCV عشان تشتغل من غير أخطاء
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ننسخ ملف المكتبات ونسطبه
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ننسخ الكود بتاعك ومجلد الأوزان (weights)
COPY . .

# بنقول للدوكر: "لما تشتغل، رن كود البايثون ده فوراً"
ENTRYPOINT ["python", "run_tracker.py"]