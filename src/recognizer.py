import numpy as np

class FaceRecognizer:
    def __init__(self, database, default_threshold: float = 0.60):
        """
        Initializes the recognizer.
        database: instance of FaceDatabase.
        default_threshold: Minimum cosine similarity required to accept a match.
                           If the best match is below this threshold, return UNKNOWN.
        """
        self.database = database
        self.default_threshold = default_threshold

    def _cosine_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """
        Computes cosine similarity between two vectors.
        Assumes both vectors are already L2-normalized.
        If they are not, we normalize them here just to be safe.
        """
        # Ensure vectors are 1D
        e1 = np.array(embed1).flatten()
        e2 = np.array(embed2).flatten()
        
        # Calculate dot product
        dot_product = np.dot(e1, e2)
        
        # Calculate norms
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        
        # Protect against division by zero
        if norm_e1 == 0 or norm_e2 == 0:
            return 0.0
            
        return float(dot_product / (norm_e1 * norm_e2))

    def identify(self, query_embedding: np.ndarray, threshold: float = None) -> dict:
        """
        Identifies a face embedding against the database.
        Returns a dictionary:
        {
            'person_id': str or 'UNKNOWN',
            'name': str or 'UNKNOWN',
            'similarity': float,
            'threshold_used': float
        }
        """
        if threshold is None:
            threshold = self.default_threshold
            
        db_data = self.database.get_all_identities()
        
        if not db_data:
            return {
                'person_id': 'UNKNOWN',
                'name': 'UNKNOWN',
                'similarity': 0.0,
                'threshold_used': threshold,
                'message': 'Database is empty'
            }
            
        best_match_id = None
        best_match_name = None
        best_similarity = -1.0 # Cosine similarity range is [-1, 1]
        
        # Compare query against all enrolled identities
        for person_id, person_data in db_data.items():
            name = person_data['name']
            enrolled_embeddings = person_data['embeddings']
            
            # Since a person can have multiple enrolled embeddings,
            # we compare against all of them and take the maximum similarity
            for enrolled_embed in enrolled_embeddings:
                sim = self._cosine_similarity(query_embedding, np.array(enrolled_embed))
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = person_id
                    best_match_name = name
                    
        # Apply Threshold for UNKNOWN rejection
        # This is a critical assignment requirement!
        if best_similarity >= threshold:
            return {
                'person_id': best_match_id,
                'name': best_match_name,
                'similarity': best_similarity,
                'threshold_used': threshold
            }
        else:
            return {
                'person_id': 'UNKNOWN',
                'name': 'UNKNOWN',
                'similarity': best_similarity,
                'threshold_used': threshold,
                'best_candidate_if_ignored_threshold': best_match_name
            }
