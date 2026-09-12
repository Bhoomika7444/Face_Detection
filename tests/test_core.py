import os
import sys
import numpy as np
import pytest
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import FaceDatabase
from src.recognizer import FaceRecognizer

def test_database_enroll_and_retrieve(tmpdir):
    """Test database enrollment and retrieval"""
    db_path = str(tmpdir)
    db = FaceDatabase(db_dir=db_path, db_name='test_db.json')
    
    # Create fake embedding
    fake_embedding = np.random.rand(512).astype(np.float32)
    
    # Enroll
    db.enroll_person(person_id='p1', name='John Doe', embedding=fake_embedding)
    
    # Reload DB
    db2 = FaceDatabase(db_dir=db_path, db_name='test_db.json')
    identities = db2.get_all_identities()
    
    assert 'p1' in identities
    assert identities['p1']['name'] == 'John Doe'
    assert len(identities['p1']['embeddings']) == 1
    
    # Enroll same person again
    db2.enroll_person(person_id='p1', name='John Doe', embedding=fake_embedding)
    assert len(db2.get_all_identities()['p1']['embeddings']) == 2
    
    # Remove person
    db2.remove_person('p1')
    assert 'p1' not in db2.get_all_identities()

def test_recognizer_cosine_similarity():
    """Test cosine similarity calculation"""
    db = FaceDatabase(db_dir='dummy', db_name='dummy.json')
    recognizer = FaceRecognizer(database=db)
    
    # Vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    
    # Identical vectors should have similarity 1.0
    sim_identical = recognizer._cosine_similarity(v1, v2)
    assert np.isclose(sim_identical, 1.0)
    
    # Orthogonal vectors should have similarity 0.0
    sim_orthogonal = recognizer._cosine_similarity(v1, v3)
    assert np.isclose(sim_orthogonal, 0.0)

def test_recognizer_identification_and_unknown(tmpdir):
    """Test identification matching and UNKNOWN rejection"""
    db_path = str(tmpdir)
    db = FaceDatabase(db_dir=db_path, db_name='test_db.json')
    recognizer = FaceRecognizer(database=db, default_threshold=0.6)
    
    # Known person
    known_emb = np.array([1.0, 0.0, 0.0] * 170 + [1.0, 1.0]) # Length 512
    db.enroll_person('p1', 'Known Person', known_emb)
    
    # 1. Exact Match
    res1 = recognizer.identify(known_emb)
    assert res1['person_id'] == 'p1'
    assert res1['name'] == 'Known Person'
    assert res1['similarity'] > 0.99
    
    # 2. UNKNOWN Rejection
    unknown_emb = np.array([0.0, 1.0, 0.0] * 170 + [0.0, 1.0])
    res2 = recognizer.identify(unknown_emb)
    assert res2['person_id'] == 'UNKNOWN'
    assert res2['name'] == 'UNKNOWN'
    assert res2['similarity'] < 0.6
