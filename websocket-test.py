import asyncio
import websockets
import json

async def test_websocket():
    websocket_url = "wss://face-recognition-backend-server.onrender.com"  # Update URL as needed
    try:
        print(f"Connecting to WebSocket at {websocket_url}...")
        async with websockets.connect(websocket_url) as websocket:
            print("Connected successfully!")

            # Send session details
            session_details = {
                "year": "5",
                "specialty": "IIR",
                "group": "G6"
            }
            print(f"Sending session details: {json.dumps(session_details)}")
            await websocket.send(json.dumps(session_details))

            # Wait for the server's response
            response = await websocket.recv()
            print(f"Received response: {response}")

            # Send a dummy face recognition message
            face_recognition_message = {"img": "base64_encoded_image_here"}
            print(f"Sending face recognition message: {json.dumps(face_recognition_message)}")
            await websocket.send(json.dumps(face_recognition_message))

            # Wait for the response
            response = await websocket.recv()
            print(f"Received face recognition response: {response}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
