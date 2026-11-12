"""kycpseudo: Streamlit frontend.

Sends scans to the engine (a Hugging Face Space) and shows each page before
and after pseudonymisation. Configure the engine address in Streamlit secrets:

    ENGINE_URL = "https://<user>-kycpseudo-engine.hf.space"

The access token is typed by the person using the app, never stored in
secrets: anyone without it gets the public, watermarked demo.
"""
from __future__ import annotations

import base64
import io
import os
import zipfile

import requests
import streamlit as st

st.set_page_config(page_title="kycpseudo: KYC document pseudonymisation", page_icon="🪪", layout="wide")


def engine_url() -> str:
    try:
        return st.secrets["ENGINE_URL"].rstrip("/")
    except Exception:
        return os.environ.get("ENGINE_URL", "http://localhost:7860").rstrip("/")


@st.cache_data(ttl=3600, show_spinner=False)
def list_samples(url: str) -> list[str]:
    r = requests.get(f"{url}/samples", timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_sample(url: str, name: str) -> bytes:
    r = requests.get(f"{url}/samples/{name}", timeout=120)
    r.raise_for_status()
    return r.content


def call_engine(url: str, files: list[tuple[str, bytes]], token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = [("files", (name, data, "application/octet-stream")) for name, data in files]
    r = requests.post(f"{url}/pseudonymise", files=payload, headers=headers, timeout=900)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Engine returned {r.status_code}: {detail}")
    return r.json()


FIELD_LABELS = {
    "surname": "Surname", "given_names": "Given names", "full_name": "Name", "third_party_name": "Name in text",
    "doc_number": "Document number", "dob": "Date of birth", "place_of_birth": "Place of birth",
    "address": "Address", "postcode": "Postcode", "account_number": "Account number", "iban": "IBAN",
}

# ------------------------------------------------------------------ sidebar
url = engine_url()
with st.sidebar:
    st.header("Settings")
    token = st.text_input("Access token (optional)", type="password",
                          help="Without a token you get the public demo: outputs are watermarked and "
                               "pseudonyms change on every run. The token returns clean output.")
    st.caption(f"Engine: `{url}`")
    try:
        health = requests.get(f"{url}/health", timeout=60).json()
        st.success("Engine online")
        st.caption(f"Up to {health['max_files']} files, {health['max_file_mb']} MB each.")
    except Exception:
        st.warning("Engine not reachable yet. A sleeping Hugging Face Space can take a minute to wake up.")
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
tab_samples, tab_upload = st.tabs(["Synthetic samples", "Upload"])
files: list[tuple[str, bytes]] = []
with tab_samples:
    st.caption("Fictional people from the fictional state of Utopia. Pick documents belonging to the same "
               "person to see one pseudonym carried across them.")
    try:
        names = list_samples(url)
        chosen = st.multiselect("Documents", names, default=names[:3], max_selections=8)
        if chosen:
            cols = st.columns(min(len(chosen), 4))
            for i, name in enumerate(chosen):
                cols[i % len(cols)].image(get_sample(url, name), caption=name, width="stretch")
        sample_files = [(n, get_sample(url, n)) for n in chosen]
    except Exception:
        sample_files = []
        st.info("Samples load once the engine is online.")
    if st.button("Pseudonymise samples", type="primary", disabled=not sample_files):
        files = sample_files

with tab_upload:
    st.warning("Upload only documents you are entitled to process. Files are processed in memory on the "
               "engine and are not stored. Without an access token, results are watermarked.")
    uploads = st.file_uploader("Scans (PNG, JPG, TIFF or PDF)", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"],
                               accept_multiple_files=True)
    if st.button("Pseudonymise uploads", type="primary", disabled=not uploads):
        files = [(u.name, u.getvalue()) for u in uploads]

# ------------------------------------------------------------------ run
if files:
    with st.spinner(f"Processing {len(files)} document(s). This takes about 5-10 seconds per page."):
        try:
            st.session_state["result"] = call_engine(url, files, token.strip())
            st.session_state["inputs"] = dict(files)
        except Exception as e:
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
            left, right = st.columns(2)
            if d["upload"] in inputs and not d["upload"].lower().endswith(".pdf"):
                left.image(inputs[d["upload"]], caption="Original", width="stretch")
            right.image(base64.b64decode(d["image_png_b64"]), caption="Pseudonymised", width="stretch")
            rows = [{"Field": FIELD_LABELS.get(f["type"], f["type"]), "Original": f["original"],
                     "Replaced with": f["surrogate"], "Person": f["entity"]} for f in d["fields"]]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
