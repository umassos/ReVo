# Frequently Asked Questions

---

## Known Issues

### DCVC-RT: encoder and decoder must run on the same GPU class

When using the `dcvcrt` codec, the sender and receiver **must run on machines
with the same GPU architecture** (e.g., both on 4070, or both on 5070).
DCVC-RT produces bitstreams that are sensitive to the floating-point behavior of
the GPU used during encoding; decoding on a different GPU class can cause
corrupted or mismatched outputs.

**Workaround:** ensure both machines are provisioned with the same GPU model
before running a `dcvcrt` session.

**Reference:** [microsoft/DCVC issue #118](https://github.com/microsoft/DCVC/issues/118)

---
