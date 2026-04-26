"""
signalling_server.py  —  ReVo

Minimal WebRTC signaling server using aiohttp WebSockets.

Peers connect to:
    ws://<host>:8080/ws/<room_id>

The server routes SDP messages between exactly two peers in a room:
  • "offer" role  — the sender
  • "answer" role — the receiver

Message flow:
    Sender                    Server                   Receiver
      │── join (offer) ──────►│                            │
      │                       │◄── join (answer) ──────────│
      │── offer (SDP) ────────►│── offer (SDP) ────────────►│
      │                       │◄── answer (SDP) ────────────│
      │◄── answer (SDP) ──────│                            │
      │         [WebRTC DataChannels established]          │
      │── bye ────────────────►│── bye ────────────────────►│

If the receiver joins after the offer has already been sent, the server
delivers the cached offer immediately so the sender does not need to retry.

Usage:
    python signalling_server.py
"""

import asyncio
from aiohttp import web, WSMsgType

# In-memory room registry.
# room_id → {"offer": ws | None, "answer": ws | None, "cached_offer": dict | None}
ROOMS: dict = {}


def _cleanup_room(room_id: str, role: str):
    """
    Remove the departing peer's WebSocket slot from the room.
    If the offer peer leaves, the cached SDP offer is also cleared because
    the sender will create a fresh offer on reconnect.
    If both peers have gone, delete the room entirely.
    """
    room = ROOMS.get(room_id)
    if not room:
        return

    room[role] = None

    if role == "offer":
        room["cached_offer"] = None

    if room.get("offer") is None and room.get("answer") is None:
        del ROOMS[room_id]
        print(f"[server] Room {room_id} deleted (both peers gone)")


async def websocket_handler(request):
    """
    Handle one WebSocket connection.

    Supported message types (JSON):
      join   – register as "offer" or "answer" in the room
      offer  – SDP offer from sender → forwarded (or cached) for receiver
      answer – SDP answer from receiver → forwarded to sender
      bye    – session teardown → forwarded to the opposite peer
    """
    ws      = web.WebSocketResponse()
    await ws.prepare(request)

    room_id = request.match_info["room"]
    role    = None  # set on "join"

    if room_id not in ROOMS:
        ROOMS[room_id] = {"offer": None, "answer": None, "cached_offer": None}

    print(f"[server] New connection for room '{room_id}'")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data     = msg.json()
                msg_type = data.get("type")

                if msg_type == "join":
                    role              = data["role"]
                    ROOMS[room_id][role] = ws
                    print(f"[server] '{role}' joined room '{room_id}'")

                    # Deliver cached offer if the receiver joined late
                    if role == "answer":
                        cached = ROOMS[room_id].get("cached_offer")
                        if cached:
                            await ws.send_json(cached)
                            print(f"[server] Delivered cached offer to receiver in room '{room_id}'")

                elif msg_type == "offer":
                    ROOMS[room_id]["cached_offer"] = data
                    target = ROOMS[room_id].get("answer")
                    if target and not target.closed:
                        await target.send_json(data)
                        print(f"[server] Relayed offer → answer in room '{room_id}'")
                    else:
                        print(f"[server] Cached offer; receiver not yet in room '{room_id}'")

                elif msg_type == "answer":
                    target = ROOMS[room_id].get("offer")
                    if target and not target.closed:
                        await target.send_json(data)
                        print(f"[server] Relayed answer → offer in room '{room_id}'")
                    else:
                        print(f"[server] Answer received but sender is gone in room '{room_id}'")

                elif msg_type == "bye":
                    # Forward to the opposite peer so it can shut down cleanly
                    other_role = "answer" if role == "offer" else "offer"
                    target     = ROOMS[room_id].get(other_role)
                    if target and not target.closed:
                        await target.send_json(data)
                        print(f"[server] Relayed bye: {role} → {other_role} in room '{room_id}'")
                    else:
                        print(f"[server] Bye from '{role}' — other peer already gone in room '{room_id}'")

            elif msg.type == WSMsgType.ERROR:
                print(f"[server] WebSocket error in room '{room_id}': {ws.exception()}")

    finally:
        print(f"[server] Connection closed: room '{room_id}', role '{role}'")
        if role is not None:
            _cleanup_room(room_id, role)

    return ws


app = web.Application()
app.add_routes([web.get("/ws/{room}", websocket_handler)])

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
