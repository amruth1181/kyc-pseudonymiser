# Deployment: engine on Hugging Face, frontend on Streamlit

```
 Browser ──> Streamlit Community Cloud ──HTTPS──> Hugging Face Docker Space
             deploy/frontend/                     deploy/engine/
             upload, token box, results           FastAPI + Tesseract + kycpseudo
```

The frontend is thin: it sends files and shows results. The engine does all the work and holds the keys.

## Access model

| Caller | Output | Keys used | Vault returned |
| --- | --- | --- | --- |
| No token (public) | Watermarked across the page, red banner | Fresh random keys per request: pseudonyms change every run and cannot be reversed | No |
| `Authorization: Bearer <ENGINE_TOKEN>` | Clean | Your persistent keys: the same person always gets the same pseudonym | Yes, encrypted |

Public requests are also rate-limited (20 per hour per IP by default). Uploads are processed in a temporary directory that is deleted before the response is sent; nothing is stored or logged except counts and timings. Uploaded filenames are never reused, because scan filenames often contain the customer's name.

## 1. Push the library to GitHub

The engine's Dockerfile installs `kycpseudo` from `github.com/amruth1181/kyc-pseudonymiser`, so the repo must be public on GitHub first.

## 2. Create the engine Space on Hugging Face

1. huggingface.co → New Space → name `kycpseudo-engine`, SDK **Docker**, hardware **CPU basic** (free), visibility **Public**.
2. Put the three files from `deploy/engine/` at the root of the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<hf-user>/kycpseudo-engine
   cp deploy/engine/{Dockerfile,app.py,README.md} kycpseudo-engine/
   cd kycpseudo-engine && git add . && git commit -m "Add kycpseudo engine" && git push
   ```
3. Space → Settings → **Variables and secrets** → add three secrets:
   ```bash
   # ENGINE_TOKEN: your access token (keep it private)
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   # KYCPSEUDO_HMAC_KEY: derives every surrogate
   python -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
   # KYCPSEUDO_VAULT_KEY: encrypts the vault
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
   Store the HMAC and vault keys somewhere safe too. Losing the HMAC key breaks consistency with earlier runs; losing the vault key makes vaults unreadable.
4. Wait for the build (about 3-5 minutes), then open `https://<hf-user>-kycpseudo-engine.hf.space/health`. It should report `"token_configured": true`. Interactive API docs are at `/docs`.

Optional variables: `MAX_FILES` (default 8), `MAX_FILE_MB` (8), `PUBLIC_REQUESTS_PER_HOUR` (20).

## 3. Deploy the frontend on Streamlit Community Cloud

1. share.streamlit.io → **Create app** → from GitHub: repo `amruth1181/kyc-pseudonymiser`, branch `main`, main file `deploy/frontend/streamlit_app.py`.
2. Advanced settings → **Secrets**:
   ```toml
   ENGINE_URL = "https://<hf-user>-kycpseudo-engine.hf.space"
   ```
   Do **not** put `ENGINE_TOKEN` in the Streamlit secrets: every visitor would then get clean output. The token is typed into the app's sidebar by whoever holds it.
3. Deploy. Streamlit installs `deploy/frontend/requirements.txt` (streamlit, requests only).

## Run both locally

```bash
pip install -e ".[pdf]" fastapi "uvicorn[standard]" python-multipart streamlit requests
# engine
cd deploy/engine && ENGINE_TOKEN=dev-token uvicorn app:app --port 7860
# frontend, in another terminal
ENGINE_URL=http://localhost:7860 streamlit run deploy/frontend/streamlit_app.py
```

## Operating notes

- **Speed:** about 5-10 seconds per page on the free CPU Space. A sleeping Space takes about a minute to wake on the first request.
- **Reproducible builds:** build with `--build-arg KYCPSEUDO_REF=<tag or commit>` (set it as a Space variable) instead of tracking `main`.
- **Real documents:** the detection rules are tuned to the synthetic templates. Before using real KYC scans, add layout rules for those document types, and run the engine as a private Space inside an approved environment.
