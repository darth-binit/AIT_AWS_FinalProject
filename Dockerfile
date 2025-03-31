FROM python:3.9-slim
LABEL authors="binit"

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy just the app.py file into /Application/app.py inside the container
COPY Application /Application

EXPOSE 8501

CMD ["streamlit", "run", "Application/app.py", "--server.port=8501", "--server.address=0.0.0.0"]