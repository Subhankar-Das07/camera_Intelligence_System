import os
import json
import logging
import cv2
import numpy as np
import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Connect to Redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    logging.info("Connected to Redis successfully.")
except redis.ConnectionError as e:
    logging.error(f"Failed to connect to Redis: {e}")
    exit(1)

# Mapping of local directories to Redis keyspaces
MIGRATIONS = [
    {"dir": "face_recognition/data", "prefix": "fr:visitor"},
    {"dir": "face_recognition/attendance_data", "prefix": "fr:attendance"}
]

for migration in MIGRATIONS:
    base_dir = migration["dir"]
    prefix = migration["prefix"]
    
    if not os.path.exists(base_dir):
        logging.info(f"Directory {base_dir} does not exist. Skipping.")
        continue

    logging.info(f"--- Migrating {base_dir} to Redis keyspace '{prefix}' ---")

    # 1. Identities
    identities_file = os.path.join(base_dir, "identities.json")
    identities = {}
    if os.path.exists(identities_file):
        try:
            with open(identities_file, "r", encoding="utf-8") as f:
                identities = json.load(f)
            r.set(f"{prefix}:identities", json.dumps(identities).encode())
            logging.info(f"Migrated {len(identities)} identities metadata.")
        except Exception as e:
            logging.error(f"Failed to load identities from {identities_file}: {e}")
    else:
        logging.info(f"No identities.json found in {base_dir}")

    # 2. Embeddings
    embeddings_file = os.path.join(base_dir, "embeddings.npz")
    if os.path.exists(embeddings_file):
        try:
            data = np.load(embeddings_file, allow_pickle=False)
            pipe = r.pipeline()
            emb_count = 0
            for person_id in data.files:
                arr = data[person_id]
                meta = {"shape": list(arr.shape), "dtype": "float32"}
                pipe.set(f"{prefix}:emb:{person_id}", np.ascontiguousarray(arr, dtype=np.float32).tobytes())
                pipe.set(f"{prefix}:embmeta:{person_id}", json.dumps(meta).encode())
                emb_count += 1
            pipe.execute()
            logging.info(f"Migrated {emb_count} embedding matrices.")
        except Exception as e:
            logging.error(f"Failed to load embeddings from {embeddings_file}: {e}")
    else:
        logging.info(f"No embeddings.npz found in {base_dir}")

    # 3. Faces (Images)
    persons_dir = os.path.join(base_dir, "persons")
    if os.path.exists(persons_dir):
        pipe = r.pipeline()
        person_count = 0
        total_crops = 0
        for person_id in os.listdir(persons_dir):
            person_path = os.path.join(persons_dir, person_id)
            if not os.path.isdir(person_path):
                continue
            
            crops = sorted([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if crops:
                person_count += 1
                total_crops += len(crops)
                for i, crop_file in enumerate(crops):
                    crop_path = os.path.join(person_path, crop_file)
                    img = cv2.imread(crop_path)
                    if img is not None:
                        ok, buf = cv2.imencode('.jpg', img)
                        if ok:
                            idx = i + 1
                            pipe.set(f"{prefix}:face:{person_id}:{idx}", buf.tobytes())
                pipe.set(f"{prefix}:face_count:{person_id}", str(len(crops)).encode())
        
        pipe.execute()
        logging.info(f"Migrated {total_crops} face images for {person_count} persons.")
    else:
        logging.info(f"No persons directory found in {base_dir}")

logging.info("Migration complete!")
