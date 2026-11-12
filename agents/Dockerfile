# Lightweight Python image for the agent workflow (separate from Airflow)
FROM python:3.12-slim

WORKDIR /app

# Install Python packages first so this layer is cached between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent code
COPY . .

# Streamlit serves the live agent graph on port 8501
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]