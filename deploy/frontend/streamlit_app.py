"""kycpseudo: Streamlit app.

Runs the whole pipeline inside the app (no separate engine) and shows each page
before and after pseudonymisation. Deploys on Streamlit Community Cloud as is:
Tesseract comes from packages.txt at the repository root, Python packages from
requirements.txt next to this file.

Optional Streamlit secrets (see .streamlit/secrets.toml.example):

    ENGINE_TOKEN          access token for clean output
    KYCPSEUDO_HMAC_KEY    base64 32-byte key for surrogates (token holders)
    KYCPSEUDO_VAULT_KEY   Fernet key for the vault (token holders)

Without the token a visitor gets the public demo: watermarked output and
throwaway keys. The token is typed into the sidebar, so only its holder gets
clean output.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import streamlit as st

# Use the library from this repository checkout, so the app always matches the code beside it.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kycpseudo import service  # noqa: E402

st.set_page_config(page_title="kycpseudo: KYC document pseudonymisation", page_icon="🪪", layout="wide")


def setting(name: str, default: str = "") -> str:
    """A value from Streamlit secrets, falling back to the environment."""
    try:
        return str(st.secrets[name])
    except Exception:
        return os.environ.get(name, default)


MAX_FILES = int(setting("MAX_FILES", "8"))
MAX_FILE_MB = int(setting("MAX_FILE_MB", "8"))
PUBLIC_PER_HOUR = int(setting("PUBLIC_REQUESTS_PER_HOUR", "20"))


@st.cache_resource(show_spinner="Preparing sample documents...")
def samples_dir() -> Path:
    return service.make_samples(Path(tempfile.gettempdir()) / "kyc-samples")


@st.cache_resource
def pipeline_lock() -> threading.Lock:
    """One pipeline run at a time across all visitors, to stay inside the host's memory."""
    return threading.Lock()


@st.cache_data(show_spinner=False)
def get_sample(name: str) -> bytes:
    return (samples_dir() / Path(name).name).read_bytes()


def allow_public_run() -> bool:
    """Per-visitor hourly limit for runs without a token."""
    now = time.time()
    hits = [t for t in st.session_state.get("hits", []) if now - t < 3600]
    allowed = len(hits) < PUBLIC_PER_HOUR
    if allowed:
        hits.append(now)
    st.session_state["hits"] = hits
    return allowed


def pseudonymise(files: list[tuple[str, bytes]], operator: bool, custom_name=None, keep_photo=False) -> dict:
    with pipeline_lock():
        return service.process(files, operator, custom_name=custom_name, keep_photo=keep_photo,
                               hmac_key_b64=setting("KYCPSEUDO_HMAC_KEY") or None,
                               vault_key=setting("KYCPSEUDO_VAULT_KEY") or None,
                               max_files=MAX_FILES, max_bytes=MAX_FILE_MB * 1024 * 1024)


FIELD_LABELS = {
    "surname": "Surname", "given_names": "Given names", "full_name": "Name", "third_party_name": "Name in text",
    "doc_number": "Document number", "dob": "Date of birth", "place_of_birth": "Place of birth",
    "address": "Address", "postcode": "Postcode", "account_number": "Account number", "iban": "IBAN",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    token = st.text_input("Access token (optional)", type="password",
                          help="Without a token you get the public demo: outputs are watermarked and "
                               "pseudonyms change on every run. The token returns clean output.")
    operator = service.token_matches(token, setting("ENGINE_TOKEN"))
    if operator:
        st.success("Token accepted: clean output.")
    elif token:
        st.error("Token not recognised: public demo mode.")
    st.subheader("Name to use (optional)")
    custom_surname = st.text_input("Surname", max_chars=41, placeholder="generated if empty")
    custom_given = st.text_input("Given names", max_chars=62, placeholder="generated if empty")
    st.caption("Used when the documents belong to one person. Document number, dates and places "
               "are always generated.")
    replace_photo = st.checkbox("Replace the passport photo", value=True,
                                help="Untick to keep the original face. A page that keeps the face is "
                                     "always watermarked, with or without a token.")
    st.caption(f"Up to {MAX_FILES} files, {MAX_FILE_MB} MB each.")
    st.divider()
    st.markdown("[Source on GitHub](https://github.com/amruth1181/kyc-pseudonymiser)")

# ------------------------------------------------------------------ header
st.title("KYC document pseudonymisation")
st.markdown(
    "Replaces personal details on scanned KYC documents with **consistent, realistic stand-ins**, "
    "instead of blacking them out. The same person gets the same pseudonym on every document "
    "in a batch, the new text is drawn in the document's own style, and every page is verified "
    "before release."
)
with st.expander("How it works"):
    st.markdown(
        "1. **Ingest:** deskew, drop hidden PDF text.\n"
        "2. **Classify:** passport, bill, statement.\n"
        "3. **Locate:** OCR and MRZ, with checksum-guided repair.\n"
        "4. **Label:** typed personal fields.\n"
        "5. **Resolve:** BM25 links mentions of the same person across documents.\n"
        "6. **Generate:** keyed surrogates, FF1 for numbers, recomputed check digits.\n"
        "7. **Render:** erase and redraw in the fitted font, size, blur and ink.\n"
        "8. **Verify:** re-read, leak search, pixel locality.\n"
        "9. **Release:** review queue, audit log, encrypted vault."
    )

# ------------------------------------------------------------------ input
tab_upload, tab_samples = st.tabs(["Upload", "Synthetic samples"])
files: list[tuple[str, bytes]] = []
with tab_samples:
    st.caption("Fictional people from the fictional state of Utopia. Pick documents belonging to the same "
               "person to see one pseudonym carried across them.")
    names = sorted(p.name for p in samples_dir().glob("*.png"))
    chosen = st.multiselect("Documents", names, default=names[:3], max_selections=MAX_FILES)
    if chosen:
        cols = st.columns(min(len(chosen), 4))
        for i, name in enumerate(chosen):
            cols[i % len(cols)].image(get_sample(name), caption=name, width="stretch")
    sample_files = [(n, get_sample(n)) for n in chosen]
    if st.button("Pseudonymise samples", type="primary", disabled=not sample_files):
        files = sample_files

with tab_upload:
    st.warning("Upload only documents you are entitled to process. Files are processed in a temporary "
               "folder that is deleted straight away; nothing is stored. Without an access token, results are watermarked.")
    uploads = st.file_uploader("Scans (PNG, JPG, TIFF or PDF)", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"],
                               accept_multiple_files=True)
    if st.button("Pseudonymise uploads", type="primary", disabled=not uploads):
        files = [(u.name, u.getvalue()) for u in uploads]

# ------------------------------------------------------------------ run
if files:
    with st.spinner(f"Processing {len(files)} document(s). This takes about 5-10 seconds per page."):
        try:
            if not operator and not allow_public_run():
                raise service.ServiceError(429, f"Limit of {PUBLIC_PER_HOUR} runs per hour reached. Try again later.")
            chosen_name = service.parse_custom_name(custom_surname, custom_given)
            st.session_state["result"] = pseudonymise(files, operator, chosen_name, keep_photo=not replace_photo)
            st.session_state["inputs"] = dict(files)
        except service.ServiceError as e:
            st.error(str(e))

res = st.session_state.get("result")
if res:
    inputs = st.session_state.get("inputs", {})
    s = res["summary"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", s["documents"])
    c2.metric("Passed checks", s["passed"])
    c3.metric("Held for review", s["sent_to_review"])
    c4.metric("People found", s["entities"])
    c5.metric("Seconds", res["seconds"])
    if res.get("custom_name") == "ignored_several_people":
        st.warning("The name you entered was not used: these documents belong to more than one person, "
                   "so names were generated.")
    if res["watermarked"]:
        st.info("Public demo: outputs are watermarked. Enter an access token in the sidebar for clean output.")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        for d in res["documents"]:
            z.writestr(d["output_name"], base64.b64decode(d["image_png_b64"]))
        if res.get("vault_b64"):
            z.writestr("vault.enc", base64.b64decode(res["vault_b64"]))
    st.download_button("Download all outputs (.zip)", archive.getvalue(), "kycpseudo_outputs.zip", "application/zip")

    for d in res["documents"]:
        ok = d["status"] == "passed"
        title = f"{'✅' if ok else '⚠️'} {d['upload']} → {d['output_name']} ({d['doc_type'].replace('_', ' ')})"
        with st.expander(title, expanded=True):
            if not ok:
                st.warning("Held for review: " + ", ".join(d["review_reasons"]))
            if d.get("watermarked") and not res["watermarked"]:
                st.warning("Watermarked: this page still shows the original photo, so it is not released clean.")
            left, right = st.columns(2)
            if d["upload"] in inputs and not d["upload"].lower().endswith(".pdf"):
                left.image(inputs[d["upload"]], caption="Original", width="stretch")
            right.image(base64.b64decode(d["image_png_b64"]), caption="Pseudonymised", width="stretch")
            rows = [{"Field": FIELD_LABELS.get(f["type"], f["type"]), "Original": f["original"],
                     "Replaced with": f["surrogate"], "Person": f["entity"]} for f in d["fields"]]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
