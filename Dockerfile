FROM python:latest

WORKDIR .

COPY requirements.txt .

COPY recommendationWords.py .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "./recommendationWords.py" ]

