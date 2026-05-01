# FAQ & Troubleshooting

- [Known Issues](#known-issues)
- [Receiver](#receiver)
- [Sender](#sender)

---

## Known Issues

### DCVC-RT: encoder and decoder must run on the same GPU class

When using the `dcvcrt` codec, the sender and receiver **must run on machines
with the same GPU architecture** (e.g., both on RTX 4070, or both on RTX 5070).
DCVC-RT produces bitstreams that are sensitive to the floating-point behavior of
the GPU used during encoding; decoding on a different GPU class can cause
corrupted or mismatched outputs.

**Workaround:** ensure both machines are provisioned with the same GPU model
before running a `dcvcrt` session.

**Reference:** [microsoft/DCVC issue #118](https://github.com/microsoft/DCVC/issues/118)

---

## Receiver

**Receiver connects but no frames arrive**
> Check that the sender is running and connected to the same signaling server.
> Verify the STUN server is reachable (needed for NAT traversal).

**`ModuleNotFoundError` for a wrapper module**
> Run from the directory containing `H265_wrapper.py` etc., or set `PYTHONPATH`.

**Saved video contains frames with artifacts**
> Ensure the receiver is using the same `CODEC` as `sender-3d.py`.

**Batch run: receiver exits before sender finishes**
> Increase `POST_RUN_COOLDOWN` in `run_sender_eval.py` to give the receiver
> more time to save its output before the next run begins.

**Batch run: output goes to `mixed/` instead of the right category**
> Ensure `CATEGORIES` in `run_receiver_eval.py` contains all category names
> used in your `trace_map.txt` (e.g. `wifi`, `cell`, `eth`).

---

## Sender

**Receiver never gets an offer**
> Confirm the signaling server is running at `ws://<server_ip>:8080/ws/demo`.
> Start the receiver before the sender sends the offer.

**`ModuleNotFoundError` for a wrapper**
> Run the script from the directory containing `H265_wrapper.py` etc., or set `PYTHONPATH`.

**Batch run skips every video (`sync failed`)**
> Ensure the receiver eval service is running and that `RECEIVER_IP` / `CONTROL_PORT`
> match between sender and receiver eval scripts.
