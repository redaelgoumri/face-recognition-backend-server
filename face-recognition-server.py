import os
import os

# Disable GPU to avoid cuDNN/cuBLAS errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# Load environment variables from the .env file
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
        response = supabase.table("Students").select("first_name, last_name, embeddings").match({
            "year": year,
            "specialty": specialty,
            "group": group
        }).execute()

        if response.data:
            # Store embeddings in memory for quick access
            session_embeddings.clear()
            for student in response.data:
                name = f"{student['first_name']} {student['last_name']}"
                session_embeddings[name] = np.array(json.loads(student['embeddings']))
            print(f"Loaded embeddings for session {year}-{specialty}-{group}: {len(session_embeddings)} students.")
        else:
            print(f"No students found for session {year}-{specialty}-{group}.")
    except Exception as e:
        print(f"Error loading session embeddings: {e}")


# WebSocket handler
async def websocket_handler(websocket):
    print("New connection established.")
    try:
        # Expect the first message to contain session details
        initial_message = await websocket.recv()
        session_data = json.loads(initial_message)
        year = session_data.get("year")
        specialty = session_data.get("specialty")
        group = session_data.get("group")

        if not year or not specialty or not group:
            await websocket.send(json.dumps({"status": False, "message": "Invalid session details"}))
            return

        # Load session embeddings
        load_session_embeddings(year, specialty, group)

        async for message in websocket:
            response = recognize_face(message)
            await websocket.send(json.dumps(response))
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed.")
        # Clear the session embeddings when the connection is closed
        session_embeddings.clear()
        print("Cleared session embeddings.")
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")


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
            return {"status": True, "message": "Recognition successful", "name": result}
        else:
            return {"status": False, "message": "No match found."}

    except Exception as e:
        return {"status": False, "message": str(e)}


# Compare the unknown face embedding to session embeddings
def compare_faces(unknown_face_embedding):
    threshold = 0.35  # Define a threshold for similarity
    best_match = None
    best_similarity = -1

    for name, embedding in session_embeddings.items():
        similarity = cosine_similarity([unknown_face_embedding], [embedding])[0][0]
        print(f"Similarity with {name}: {similarity}")

        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = name

    return best_match  # Return the name if the match is found, otherwise None


# Main server
async def main():
    server = await websockets.serve(websocket_handler, "0.0.0.0", 8765)
    print("Server started at ws://0.0.0.0:8765")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
