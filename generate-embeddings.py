import numpy as np
from supabase import create_client
from deepface import DeepFace
import json
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# Precompute embeddings
def generate_embeddings():
    bucket_name = "studentinformation"
    students_table = "Students"

    # List all images in the bucket
    files = supabase.storage.from_(bucket_name).list(path="")
    print(f"Found {len(files)} files in the bucket.")

    for file in files:
        try:
            # Parse image name to extract student details
            filename = file['name']
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                continue

            parts = filename.split('-')
            if len(parts) < 5:
                print(f"Skipping invalid filename: {filename}")
                continue

            year, specialty, group, last_name, first_name_ext = parts
            first_name = first_name_ext.split('.')[0]  # Remove file extension

            # Generate the signed URL
            signed_url = supabase.storage.from_(bucket_name).create_signed_url(file['name'], 7 * 24 * 60 * 60).get(
                "signedURL")
            if not signed_url:
                print(f"Failed to generate signed URL for {filename}")
                continue

            # Generate embedding
            print(f"Processing {filename}...")
            embedding = DeepFace.represent(img_path=signed_url, model_name="VGG-Face")[0]["embedding"]

            # Update the database
            student_data = {
                "year": year,
                "specialty": specialty,
                "group": group,
                "last_name": last_name,
                "first_name": first_name,
                "isMale" : isMale
            }
            embedding_json = json.dumps(embedding)

            # Check if student exists in the table
            response = supabase.table(students_table).select("*").match(student_data).execute()
            if response.data:
                # Update the embedding if student exists
                print(f"Updating embedding for {last_name} {first_name}")
                supabase.table(students_table).update({"embeddings": embedding_json}).match(student_data).execute()
            else:
                # Insert new student record if not exists
                print(f"Inserting new student: {last_name} {first_name}")
                student_data["embeddings"] = embedding_json
                supabase.table(students_table).insert(student_data).execute()

        except Exception as e:
            print(f"Error processing {file['name']}: {e}")


# Run the script
if __name__ == "__main__":
    generate_embeddings()
