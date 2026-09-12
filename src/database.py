import os
import json
import numpy as np

class FaceDatabase:
    def __init__(self, db_dir: str = 'embeddings', db_name: str = 'database.json'):
        """
        Initializes the database.
        Embeddings are stored in a JSON file containing lists of floats.
        """
        self.db_dir = db_dir
        self.db_path = os.path.join(db_dir, db_name)
        self.data = {} # Format: {'person_id': {'name': 'John Doe', 'embeddings': [[...], [...]]}}
        
        # Ensure database directory exists
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            
        self.load_db()

    def load_db(self):
        """Loads database from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                self.data = {}
        else:
            self.data = {}

    def save_db(self):
        """Saves database to disk."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Error saving database: {e}")

    def enroll_person(self, person_id: str, name: str, embedding: np.ndarray):
        """
        Enrolls a person. If person_id already exists, appends the new embedding.
        embedding should be a 512-D numpy array.
        """
        embedding_list = embedding.tolist()
        
        if person_id not in self.data:
            self.data[person_id] = {
                'name': name,
                'embeddings': [embedding_list]
            }
        else:
            self.data[person_id]['embeddings'].append(embedding_list)
            
        self.save_db()

    def get_all_identities(self) -> dict:
        """Returns the current database."""
        return self.data

    def remove_person(self, person_id: str):
        """Removes a person from the database."""
        if person_id in self.data:
            del self.data[person_id]
            self.save_db()
            
    def clear_database(self):
        """Clears all entries from the database."""
        self.data = {}
        self.save_db()
