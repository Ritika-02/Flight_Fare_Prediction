From python:3.10-slim-buster
WORKDIR /service 
COPY requirements.txt .
COPY . ./                 
RUN pip install -r requirements.txt
ENTRYPOINT ["python3","app.py"]