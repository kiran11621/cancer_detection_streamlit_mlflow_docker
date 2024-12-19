# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy app files
COPY app/ /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]
