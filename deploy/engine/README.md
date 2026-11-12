---
title: kycpseudo engine
emoji: 🪪
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: API that pseudonymises scanned KYC documents
---

# kycpseudo engine

HTTP API for [kycpseudo](https://github.com/amruth1181/kyc-pseudonymiser). Upload scans to `POST /pseudonymise` and get each page back with personal details replaced by consistent stand-ins.

- Anonymous requests get **watermarked** output and per-request throwaway keys.
- Requests with `Authorization: Bearer <ENGINE_TOKEN>` get clean output and the encrypted vault.
- Nothing is stored. Use the synthetic samples at `GET /samples`, not real identity documents.

API docs: `/docs` on this Space.
