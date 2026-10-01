FROM python:3.12-slim

WORKDIR /opt/dimebench
COPY . /opt/dimebench
RUN python -m pip install --no-cache-dir .

VOLUME ["/results"]
ENTRYPOINT ["dimebench"]
CMD ["smoke", "--output-dir", "/results"]
