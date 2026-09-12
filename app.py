import streamlit as st
import os
from PIL import Image
import numpy as np

# Set page config FIRST, before any other Streamlit commands
st.set_page_config(page_title="Face Recognition System", page_icon="👤", layout="wide")

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.database import FaceDatabase
from src.recognizer import FaceRecognizer
from src.utils import pil_to_cv2, cv2_to_pil, draw_faces

# Initialize ML components in session state so they aren't reloaded every interaction
@st.cache_resource
def load_models():
    detector = FaceDetector(device='cpu')
    embedder = FaceEmbedder(device='cpu')
    return detector, embedder

@st.cache_resource
def load_database():
    return FaceDatabase(db_dir='embeddings', db_name='database.json')

def main():
    st.title("Face Recognition Identification System")
    st.markdown("Code Nimbus Solutions - AI/ML Intern Assessment")

    detector, embedder = load_models()
    db = load_database()
    
    tab1, tab2, tab3 = st.tabs(["👤 Enroll Person", "🔍 Identify Face", "🗄️ Database Manager"])

    # --- TAB 1: ENROLLMENT ---
    with tab1:
        st.header("Enroll New Person")
        person_id = st.text_input("Person ID (e.g. employee123)", "")
        person_name = st.text_input("Full Name", "")
        uploaded_file = st.file_uploader("Upload Face Image for Enrollment", type=['jpg', 'jpeg', 'png'])

        if st.button("Enroll Person"):
            if not person_id or not person_name:
                st.error("Please provide both ID and Name.")
            elif not uploaded_file:
                st.error("Please upload an image.")
            else:
                image = Image.open(uploaded_file).convert('RGB')
                faces = detector.detect_faces(image)
                
                if len(faces) == 0:
                    st.error("No face detected in the image. Please try another image.")
                elif len(faces) > 1:
                    st.warning("Multiple faces detected! Please use an image with only one clear face for enrollment.")
                else:
                    face = faces[0]
                    # Generate embedding
                    embedding = embedder.get_embedding(face['face_tensor'])
                    
                    # Store in database
                    db.enroll_person(person_id, person_name, embedding)
                    st.success(f"Successfully enrolled {person_name} (ID: {person_id})!")
                    
                    # Preview cropped face
                    st.image(detector.crop_face(image, face['box']), caption="Enrolled Face", width=150)

    # --- TAB 2: IDENTIFICATION ---
    with tab2:
        st.header("Identify Faces")
        threshold = st.slider("Similarity Threshold (UNKNOWN Rejection)", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
        recognizer = FaceRecognizer(database=db, default_threshold=threshold)
        
        identify_file = st.file_uploader("Upload Image to Identify", type=['jpg', 'jpeg', 'png'], key="identify")
        
        if identify_file:
            image = Image.open(identify_file).convert('RGB')
            faces = detector.detect_faces(image)
            
            if len(faces) == 0:
                st.warning("No face detected.")
                st.image(image, use_column_width=True)
            else:
                st.info(f"Detected {len(faces)} face(s).")
                
                faces_draw_data = []
                results_data = []
                
                for face in faces:
                    # Generate embedding
                    embedding = embedder.get_embedding(face['face_tensor'])
                    
                    # Identify
                    result = recognizer.identify(embedding, threshold=threshold)
                    
                    # Determine color and label
                    if result['person_id'] == 'UNKNOWN':
                        color = (255, 0, 0) # Red for unknown in RGB
                        label = "UNKNOWN"
                    else:
                        color = (0, 255, 0) # Green for known in RGB
                        label = result['name']
                        
                    faces_draw_data.append({
                        'box': face['box'],
                        'label': label,
                        'score': result['similarity'],
                        'color': color # OpenCV uses BGR, but we convert back to PIL below so it doesn't matter if we adapt utils or use RGB here and flip.
                        # Wait, utils draw_faces expects BGR image and BGR color.
                        # Red in BGR is (0, 0, 255). Green is (0, 255, 0).
                    })
                    
                    # Fix colors for BGR utils
                    faces_draw_data[-1]['color'] = (0, 0, 255) if label == "UNKNOWN" else (0, 255, 0)
                    
                    results_data.append(result)
                
                # Draw on image
                cv2_img = pil_to_cv2(image)
                cv2_img_drawn = draw_faces(cv2_img, faces_draw_data)
                final_image = cv2_to_pil(cv2_img_drawn)
                
                st.image(final_image, caption="Identification Results", use_column_width=True)
                
                st.subheader("Detailed Results")
                for i, res in enumerate(results_data):
                    if res['name'] == 'UNKNOWN':
                        st.error(f"Face {i+1}: UNKNOWN (Max Similarity: {res['similarity']:.2f}, Threshold: {threshold}) - Potential Candidate: {res.get('best_candidate_if_ignored_threshold', 'None')}")
                    else:
                        st.success(f"Face {i+1}: {res['name']} (Similarity: {res['similarity']:.2f})")

    # --- TAB 3: DATABASE MANAGER ---
    with tab3:
        st.header("Enrolled Identities")
        identities = db.get_all_identities()
        
        if not identities:
            st.info("Database is empty. Enroll some people first.")
        else:
            for pid, data in identities.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{data['name']}** (ID: {pid}) - {len(data['embeddings'])} embedding(s)")
                with col2:
                    if st.button("Remove", key=f"del_{pid}"):
                        db.remove_person(pid)
                        st.experimental_rerun()
                        
            if st.button("Clear All Database", type="primary"):
                db.clear_database()
                st.experimental_rerun()

if __name__ == "__main__":
    main()
