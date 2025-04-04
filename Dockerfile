FROM python:3.9-slim
LABEL authors="binit"

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy just the app.py file into /Project_Folder/app.py inside the container
COPY Project_Folder /Application

# Debug: Show what got copy here
RUN ls -al /Application

# Expose port 8501 (the default for Streamlit)
EXPOSE 8501

# Start Streamlit (using shell form so environment variables like PORT can be expanded if needed)
CMD streamlit run /Application/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false