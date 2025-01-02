import sys
sys.path.append('./generate_report.py')

from generate_report import generate_attendance_excel
import os
import asyncio
import uuid
import websockets
import io
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import base64
from supabase import create_client, Client
from dotenv import load_dotenv
import json
import websockets.exceptions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

# Load environment variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Store embeddings per session
session_embeddings = {}

# Load students' embeddings for the session
def load_session_embeddings(year, specialty, group):
    session_key = f"{year}-{specialty}-{group}"
    try:
        response = supabase.table("Students").select("id, first_name, last_name, embeddings").match({
            "year": year,
            "specialty": specialty,
            "group": group
        }).execute()

        if response.data:
            session_embeddings[session_key] = {}  # Initialize for the session
            for student in response.data:
                name = f"{student['first_name']} {student['last_name']}"
                session_embeddings[session_key][name] = {
                    "id": student['id'],
                    "embedding": np.array(json.loads(student['embeddings']))
                }
            print(f"Loaded embeddings for {session_key}: {len(session_embeddings[session_key])} students.", flush=True)
        else:
            print(f"No students found for {session_key}.", flush=True)
    except Exception as e:
        print(f"Error loading embeddings: {e}", flush=True)


async def websocket_handler(websocket):
    print("New connection established.", flush=True)
    present_students = set()
    session_key = ""
    session_data = {}
    session_id = None
    student_id = None

    try:
        # Receive initial session data including session_id and student_id
        initial_message = await websocket.recv()
        session_data = json.loads(initial_message)

        print(f"<DEBUG> Incoming WebSocket data: {session_data}", flush=True)

        # Extract session_id and student_id
        session_id = session_data.get("session_id", str(uuid.uuid4()))
        student_id = session_data.get("student_id")

        if not student_id:
            print("<ERROR> student_id not provided. Skipping embedding save.", flush=True)

        # Handle embedding generation
        if session_data.get("type") == "embedding":
            response = generate_embedding(session_data.get("img"))

            print(f"<DEBUG> Saving embedding for student {student_id}...", flush=True)
            embedding_data = response["embedding"]  # Ensure embeddings are JSON serializable

            # Update Supabase Students table
            supabase.table("Students") \
                .update({"embeddings": embedding_data}) \
                .eq("id", student_id) \
                .execute()

            print(f"<DEBUG> Embedding saved successfully for student {student_id}.", flush=True)
        else:
                print("<ERROR> Embedding generation failed or student_id missing. Skipping save.", flush=True)

        # Handle scan type request (Default path)
        session_data.setdefault("type", "scan")

        # Extract fields for scan sessions
        year = session_data.get("year")
        specialty = session_data.get("specialty")
        group = session_data.get("group")

        # Create session key for scan
        if year and specialty and group:
            session_key = f"{year}-{specialty}-{group}"
            load_session_embeddings(year, specialty, group)

        # Acknowledge session initialization for scans
        await websocket.send(json.dumps({
            "status": True,
            "message": "Session initialized",
            "type": session_data["type"]
        }))

        # Listen for recognition results
        async for message in websocket:
            if session_data.get("type") == "scan":
                response = recognize_face(message, session_key)
                if response["status"]:
                    present_students.add(json.dumps(response))

                await websocket.send(json.dumps(response))
                print(f"Sent to client: {response}", flush=True)

    except websockets.exceptions.ConnectionClosed:
        print(f"Session {session_key} closed.", flush=True)

    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)

    finally:
        # Remove embeddings when session ends (if applicable)
        if session_key and session_key in session_embeddings:
            del session_embeddings[session_key]

        # Fetch teacher name only if it's a scan session
        if session_data.get("type") == "scan" and 'teacher_name' not in session_data:
            try:
                session_details = supabase.table("Sessions") \
                    .select("teacher") \
                    .eq("id", session_id) \
                    .single() \
                    .execute()

                if session_details.data:
                    teacher_id = session_details.data['teacher']
                    teacher_info = supabase.table("User") \
                        .select("first_name, last_name") \
                        .eq("id", teacher_id) \
                        .single() \
                        .execute()

                    if teacher_info.data:
                        session_data[
                            'teacher_name'] = f"{teacher_info.data['first_name']} {teacher_info.data['last_name']}"
                    else:
                        session_data['teacher_name'] = "Unknown"
            except Exception as e:
                print(f"<ERROR> Failed to fetch teacher data: {e}", flush=True)

        # Generate attendance report for scan sessions
        if session_data.get("type") == "scan":
            all_students_response = supabase.table("Students") \
                .select("id, first_name, last_name, year, specialty, group") \
                .match({
                "year": year,
                "specialty": specialty,
                "group": group
            }) \
                .execute()

            all_students = all_students_response.data if all_students_response.data else []

            # Mark attendance (Present/Absent)
            for student in all_students:
                student["status"] = "Absent"
                for present in present_students:
                    parsed_present = json.loads(present)
                    if student['id'] == parsed_present['id']:
                        student["status"] = "Present"
                        break

            # Generate the attendance report
            generate_attendance_excel(session_data, all_students)

            print(f"Attendance report generated for session {session_id}.", flush=True)
            print(f"<ALL STUDENTS ATTENDANCE> {json.dumps(all_students, indent=2)}", flush=True)


# Generate embedding for new students
def generate_embedding(base64_image):
    try:
        print("<DEBUG> Starting embedding generation...", flush=True)

        # Step 1: Decode the base64 image
        image_data = base64.b64decode(base64_image)
        print("<DEBUG> Base64 image decoded successfully.", flush=True)

        # Step 2: Convert to PIL Image and prepare for DeepFace
        pil_image = Image.open(io.BytesIO(image_data))
        print("<DEBUG> PIL Image created successfully.", flush=True)

        # Step 3: Generate embedding using DeepFace
        embedding_result = DeepFace.represent(img_path=np.array(pil_image), model_name="VGG-Face")

        # Ensure the result contains data
        if not embedding_result or "embedding" not in embedding_result[0]:
            raise Exception("Embedding result is empty or invalid.")

        embedding = embedding_result[0]["embedding"]
        print(f"<DEBUG> Embedding generated successfully. Sample: {embedding[:5]}... (truncated)", flush=True)

        # Step 4: Return embedding
        return {"status": True, "embedding": json.dumps(embedding)}

    except Exception as e:
        print(f"<ERROR> Failed to generate embedding: {e}", flush=True)
        return {"status": False, "message": str(e)}




def recognize_face(message, session_key):
    try:
        request = json.loads(message)
        base64_image = request.get("img", "")
        if not base64_image:
            return {"status": False, "message": "No image provided."}

        image_data = base64.b64decode(base64_image)
        pil_image = Image.open(io.BytesIO(image_data))

        try:
            unknown_face_embedding = DeepFace.represent(img_path=np.array(pil_image), model_name="VGG-Face")[0]["embedding"]
        except Exception:
            return {"status": None, "message": "No face detected."}

        result = compare_faces(np.array(unknown_face_embedding), session_key)

        if result:
            return {"status": True, "name": result["name"], "id": result["id"]}
        else:
            return {"status": False, "message": "No match found."}

    except Exception as e:
        return {"status": False, "message": str(e)}


def compare_faces(unknown_face_embedding, session_key):
    threshold = 0.35
    best_match = None
    best_similarity = -1

    embeddings = session_embeddings.get(session_key, {})
    for name, data in embeddings.items():
        similarity = cosine_similarity([unknown_face_embedding], [data["embedding"]])[0][0]
        print(f"Similarity with {name}: {similarity}", flush=True)

        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = {"name": name, "id": data["id"]}

    return best_match

async def main():
    port = int(os.getenv("PORT", 8765))
    server = await websockets.serve(websocket_handler, "0.0.0.0", port)
    print(f"Server started at ws://0.0.0.0:{port}", flush=True)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
