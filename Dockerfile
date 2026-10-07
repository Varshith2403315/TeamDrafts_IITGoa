FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "python -m pytest -q && python scripts/run_study.py && python scripts/make_figures.py"]
