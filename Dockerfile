FROM python:3.8-slim

# נגדיר WORKDIR
WORKDIR /app

# נתקין pip ואת כל התלויות
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# התקנת חבילות Python הנדרשות
RUN pip install --no-cache-dir \
        numpy \
        scipy \
        matplotlib \
        pandas \
        tqdm

# נעתיק את קובץ הסקריפט אל התמונה
COPY simulations.py .

# ברירת מחדל: נריץ את הסקריפט
CMD ["python3", "simulation.py"]