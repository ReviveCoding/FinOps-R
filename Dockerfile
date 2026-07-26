FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/finops-r-matplotlib

WORKDIR /app

RUN groupadd --system finopsr && useradd --system --gid finopsr --create-home finopsr

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY configs ./configs
COPY data/fixtures ./data/fixtures
RUN mkdir -p /app/artifacts /tmp/finops-r-matplotlib \
    && chown -R finopsr:finopsr /app /tmp/finops-r-matplotlib

USER finopsr

ENTRYPOINT ["finops-r"]
CMD ["--help"]
