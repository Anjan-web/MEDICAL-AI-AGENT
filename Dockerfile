FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

COPY --chown=user . .

RUN chmod +x startup.sh

EXPOSE 7860

CMD ["bash", "startup.sh"]