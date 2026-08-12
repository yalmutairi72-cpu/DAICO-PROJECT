FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LANGCHAIN_TRACING_V2="true"
ENV LANGCHAIN_PROJECT="HR-Agent-Capstone"

CMD ["python", "agent.py"]
