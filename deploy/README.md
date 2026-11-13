# Deployment: Streamlit Community Cloud

```
 Browser ──> Streamlit Community Cloud
             deploy/frontend/streamlit_app.py
             upload, token box, results, and the whole pipeline (Tesseract + kycpseudo)
```

The Streamlit app runs the pipeline itself, so one free deployment is all it needs. It imports `kycpseudo` straight from `src/` in this repository, so the live app always matches the code on the deployed branch.

## Access model

| Visitor | Output | Keys used | Vault returned |
| --- | --- | --- | --- |
| No token (public) | Watermarked across the page, red banner | Fresh random keys per run: pseudonyms change every run and cannot be reversed | No |
| Types `ENGINE_TOKEN` in the sidebar | Clean | Your persistent keys: the same person always gets the same pseudonym | Yes, encrypted, in the download |

**Choosing the name.** With the token accepted, the sidebar shows optional *Surname* and *Given names* boxes. If both are filled and the documents belong to one person, that name is used instead of a generated one, on the page and in the MRZ. Only the name can be chosen: document number, dates, places and the photo are always generated, and a passport whose photo could not be replaced is watermarked when a chosen name is on it. Public visitors do not get this option.

Public runs are limited to 20 per hour per browser session, and only one run executes at a time across all visitors. Uploads are processed in a temporary directory that is deleted before the result is shown; nothing is stored. Uploaded filenames are never reused, because scan filenames often contain the customer's name.

## Deploy

1. Push this repository to GitHub (`amruth1181/kyc-pseudonymiser`).
2. share.streamlit.io → **Create app** → from GitHub: repo `amruth1181/kyc-pseudonymiser`, branch `main`, main file `deploy/frontend/streamlit_app.py`.
3. Advanced settings → Python **3.12**, and paste three **Secrets**:
   ```toml
   ENGINE_TOKEN = "..."
   KYCPSEUDO_HMAC_KEY = "..."
   KYCPSEUDO_VAULT_KEY = "..."
   ```
   Generate the values with:
   ```bash
   # ENGINE_TOKEN: your access token (keep it private)
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   # KYCPSEUDO_HMAC_KEY: derives every surrogate
   python3 -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
   # KYCPSEUDO_VAULT_KEY: encrypts the vault (any 32 random bytes, url-safe base64)
   python3 -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
   ```
   Store the HMAC and vault keys somewhere safe too. Losing the HMAC key breaks consistency with earlier runs; losing the vault key makes vaults unreadable. The secrets are optional: without them the app runs as the public demo only.
4. Deploy. Streamlit installs Tesseract from `packages.txt` at the repository root and the Python packages from `deploy/frontend/requirements.txt`.

Optional secrets: `MAX_FILES` (default 8), `MAX_FILE_MB` (8), `PUBLIC_REQUESTS_PER_HOUR` (20).

## Run locally

```bash
brew install tesseract            # or: apt-get install tesseract-ocr
pip install -r deploy/frontend/requirements.txt
ENGINE_TOKEN=dev-token streamlit run deploy/frontend/streamlit_app.py
```

## Optional: the HTTP engine

`deploy/engine/` holds the same service as a FastAPI app with a Dockerfile, for when other programs need to call it over HTTP. It suits any Docker host; on Hugging Face, Docker Spaces need a paid plan. The Streamlit app does not use it.

```bash
pip install -e ".[pdf]" fastapi "uvicorn[standard]" python-multipart
cd deploy/engine && ENGINE_TOKEN=dev-token uvicorn app:app --port 7860
```

It reads the same `ENGINE_TOKEN`, `KYCPSEUDO_HMAC_KEY` and `KYCPSEUDO_VAULT_KEY` from the environment, takes the token as `Authorization: Bearer <ENGINE_TOKEN>`, and rate-limits public requests per IP.

## Operating notes

- **Speed:** about 10 seconds per page. An app that has gone to sleep takes a moment to wake on the first visit.
- **Memory:** the free tier is small, which is why runs are queued one at a time. It suits a demo, not bulk processing.
- **Real documents:** the detection rules are tuned to the synthetic templates, with extra handling for bilingual passport layouts. Before using real KYC scans, add layout rules for those document types, and run the service inside an approved, private environment.
