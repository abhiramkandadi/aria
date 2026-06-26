"""
ARIA WebSocket bridge — Stage 1: echo server
Full inference pipeline added in Stage 2.
"""
import asyncio
import json
import websockets

HOST = "0.0.0.0"
PORT = 8765

async def handle_client(websocket):
    client_addr = websocket.remote_address
    print(f"[bridge] Connected: {client_addr}")
    try:
        async for message in websocket:
            if isinstance(message, str):
                payload = json.loads(message)
                print(f"[bridge] Received: {payload.get('type', 'unknown')}")
                await websocket.send(json.dumps({
                    "status": "ok",
                    "echo": payload,
                    "message": "ARIA bridge connected — Stage 1 echo"
                }))
            else:
                print(f"[bridge] Binary frame: {len(message)} bytes")
                await websocket.send(json.dumps({"status": "ok", "bytes_received": len(message)}))
    except websockets.exceptions.ConnectionClosed:
        print(f"[bridge] Disconnected: {client_addr}")

async def main():
    print(f"[bridge] ARIA bridge starting on ws://{HOST}:{PORT}")
    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
