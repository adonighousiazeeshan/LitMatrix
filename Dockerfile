# 1. Use Python as the base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# 4. Display logs immediately
ENV PYTHONUNBUFFERED=1

# 5. Copy requirements first
COPY requirements.txt .

# 6. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy project files
COPY . .

# 8. Streamlit port
EXPOSE 8501

# 9. Start LitMatrix
CMD ["streamlit", "run", "app1.py", "--server.address=0.0.0.0", "--server.port=8501"]