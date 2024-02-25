FROM python:3.10

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

COPY src/ /src/
COPY data/ /data/
COPY db/ /db/
COPY images/ /images/

WORKDIR /src

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "Chat_1.py", "--server.port=8080", "--server.address=0.0.0.0"]
