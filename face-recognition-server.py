import os

# Disable GPU to avoid cuDNN/cuBLAS errors
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import asyncio
import websockets
import json
import io
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import base64
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from the .env files
load_dotenv()

# Initialize Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Global temporary storage for session embeddings
session_embeddings = {}


# Load students' embeddings for the session
def load_session_embeddings(year, specialty, group):
    try:
        # Query Supabase for students in the specified year, specialty, and group
        response = supabase.table("Students").select("id, first_name, last_name, embeddings").match({
            "year": year,
            "specialty": specialty,
            "group": group
        }).execute()

        if response.data:
            # Store embeddings and user IDs in memory for quick access
            session_embeddings.clear()
            for student in response.data:
                name = f"{student['first_name']} {student['last_name']}"
                session_embeddings[name] = {
                    "id": student['id'],  # Store user ID
                    "embedding": np.array(json.loads(student['embeddings']))
                }
            print(f"Loaded embeddings for session {year}-{specialty}-{group}: {len(session_embeddings)} students.", flush=True)
        else:
            print(f"No students found for session {year}-{specialty}-{group}.", flush=True)
    except Exception as e:
        print(f"Error loading session embeddings: {e}", flush=True)



# WebSocket handler
async def websocket_handler(websocket):
    print("New connection established.", flush=True)
    try:
        # Expect the first message to contain session details
        initial_message = await websocket.recv()
        print(f"Received initial message: {initial_message}", flush=True)

        try:
            session_data = json.loads(initial_message)
        except json.JSONDecodeError as e:
            error_message = {"status": False, "message": "Invalid JSON format"}
            print(f"JSON Decode Error: {e}. Sending: {error_message}", flush=True)
            await websocket.send(json.dumps(error_message))
            return

        # Extract session details
        year = session_data.get("year")
        specialty = session_data.get("specialty")
        group = session_data.get("group")
        print(f"Parsed session details: year={year}, specialty={specialty}, group={group}", flush=True)

        if not year or not specialty or not group:
            error_message = {"status": False, "message": "Invalid session details"}
            print(f"Invalid session details: {session_data}. Sending: {error_message}", flush=True)
            await websocket.send(json.dumps(error_message))
            return

        # Load session embeddings
        print(f"Loading embeddings for year={year}, specialty={specialty}, group={group}", flush=True)
        load_session_embeddings(year, specialty, group)
        print("Embeddings loaded successfully.", flush=True)

        # Notify client of successful initialization
        success_message = {"status": True, "message": "Session initialized"}
        print(f"Sending: {success_message}", flush=True)
        await websocket.send(json.dumps(success_message))

        # Process subsequent messages for face recognition
        async for message in websocket:
            print(f"Received face recognition message: <IMAGE>", flush=True)

            try:
                response = recognize_face(message)
                print(f"Sending face recognition response: {response}", flush=True)
                await websocket.send(json.dumps(response))
            except Exception as e:
                error_message = {"status": False, "message": f"Error processing message: {str(e)}"}
                print(f"Error: {str(e)}. Sending: {error_message}", flush=True)
                await websocket.send(json.dumps(error_message))

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed.", flush=True)
    except Exception as e:
        print(f"Unexpected WebSocket Error: {str(e)}", flush=True)
    finally:
        # Clear the session embeddings when the connection is closed or an error occurs
        session_embeddings.clear()
        print("Cleared session embeddings.", flush=True)


# Recognize face from the incoming image
def recognize_face(message):
    try:
        # Decode the incoming JSON message
        request = json.loads(message)
        base64_image = request.get("img", "")
        if not base64_image:
            return {"status": False, "message": "No image provided."}

        # Convert Base64 image to PIL image
        image_data = base64.b64decode(base64_image)
        pil_image = Image.open(io.BytesIO(image_data))

        # Detect and represent faces in the image
        try:
            unknown_face_embedding = DeepFace.represent(img_path=np.array(pil_image), model_name="VGG-Face")[0]["embedding"]
        except Exception as e:
            return {"status": None, "message": "No face detected in the image."}

        # Compare with stored session embeddings
        result = compare_faces(np.array(unknown_face_embedding))

        if result:
            return {"status": True, "message": "Recognition successful", "name": result["name"], "id": result["id"]}
        else:
            return {"status": False, "message": "No match found."}

    except Exception as e:
        return {"status": False, "message": str(e)}



# Compare the unknown face embedding to session embeddings
def compare_faces(unknown_face_embedding):
    threshold = 0.35  # Define a threshold for similarity
    best_match = None
    best_similarity = -1

    for name, data in session_embeddings.items():
        similarity = cosine_similarity([unknown_face_embedding], [data["embedding"]])[0][0]
        print(f"Similarity with {name}: {similarity}", flush=True)

        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = {"name": name, "id": data["id"]}  # Include user ID

    return best_match  # Return the match if found, otherwise None


# Main server
async def main():
    port = int(os.getenv("PORT", 8765))  # Default to 8765 for local testing
    server = await websockets.serve(websocket_handler, "0.0.0.0", port)
    print("Server started at ws://0.0.0.0:8765", flush=True)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
