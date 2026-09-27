FROM python:3.11-slim

WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY . .

RUN pip install --no-cache-dir \
    streamlit \
    joblib \
    matplotlib \
    numpy \
    pandas \
    rich \
    scikit-learn \
    seaborn

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py"]
