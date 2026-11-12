# kycpseudo engine for a Hugging Face Docker Space (listens on 7860).
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng fonts-dejavu-core git \
 && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# The engine installs the library from GitHub. Pin a tag or commit for reproducible builds.
ARG KYCPSEUDO_REF=main
RUN pip install --no-cache-dir --user \
      "kycpseudo[pdf] @ git+https://github.com/amruth1181/kyc-pseudonymiser.git@${KYCPSEUDO_REF}" \
      fastapi==0.115.* "uvicorn[standard]==0.32.*" python-multipart==0.0.*

COPY --chown=user app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
