import sqlite3
import os
import cv2
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleDatabase:
    """
    Manages the SQLite database and local image storage for the vehicle recognition pipeline.
    Ensures thread-safe operations when recording visits and storing snapshots.
    """
    
    def __init__(self, db_path: str = "storage/vehicle_intelligence.db"):
        """
        Initialize the VehicleDatabase, ensuring storage directories and DB schema exist.
        
        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.storage_dir = os.path.dirname(db_path) or "storage"
        self.snapshots_dir = os.path.join(self.storage_dir, "snapshots")
        self.plate_crops_dir = os.path.join(self.storage_dir, "plate_crops")
        
        self._ensure_directories()
        
        # Thread-safe lock for database operations
        self.lock = threading.Lock()
        
        # Initialize database connection
        # check_same_thread=False allows multiple threads to share the connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _ensure_directories(self) -> None:
        """Create necessary storage directories if they do not exist."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            os.makedirs(self.snapshots_dir, exist_ok=True)
            os.makedirs(self.plate_crops_dir, exist_ok=True)
            logger.info(f"Storage directories verified/created at: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create storage directories: {e}")
            raise

    def _init_db(self) -> None:
        """Initialize the SQLite database schema for vehicles and visit history."""
        with self.lock:
            try:
                cursor = self.conn.cursor()
                
                # Create vehicles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vehicles (
                        plate_number TEXT PRIMARY KEY,
                        total_visits INTEGER DEFAULT 1,
                        first_seen DATETIME,
                        last_seen DATETIME
                    )
                """)
                
                # Create visit_history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS visit_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT,
                        visit_number INTEGER,
                        timestamp DATETIME,
                        snapshot_path TEXT,
                        plate_crop_path TEXT,
                        ocr_confidence REAL,
                        FOREIGN KEY (plate_number) REFERENCES vehicles(plate_number)
                    )
                """)
                
                self.conn.commit()
                logger.info("Database schema initialized successfully.")
            except sqlite3.Error as e:
                logger.error(f"Database initialization error: {e}")
                raise

    def record_visit(self, plate_number: str, snapshot_img: np.ndarray, plate_crop_img: np.ndarray, confidence: float) -> int:
        """
        Record a vehicle visit, save images to disk, and update the database.
        
        Args:
            plate_number (str): The recognized license plate number.
            snapshot_img (np.ndarray): The full snapshot image.
            plate_crop_img (np.ndarray): The cropped license plate image.
            confidence (float): The OCR confidence score.
            
        Returns:
            int: The total number of visits for this vehicle.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        
        with self.lock:
            try:
                cursor = self.conn.cursor()
                
                # Check if vehicle exists
                cursor.execute("SELECT total_visits FROM vehicles WHERE plate_number = ?", (plate_number,))
                row = cursor.fetchone()
                
                if row:
                    visit_num = row['total_visits'] + 1
                    cursor.execute("""
                        UPDATE vehicles 
                        SET total_visits = ?, last_seen = ? 
                        WHERE plate_number = ?
                    """, (visit_num, now, plate_number))
                else:
                    visit_num = 1
                    cursor.execute("""
                        INSERT INTO vehicles (plate_number, total_visits, first_seen, last_seen) 
                        VALUES (?, ?, ?, ?)
                    """, (plate_number, visit_num, now, now))
                    
                # Define image paths
                snapshot_filename = f"{plate_number}_visit{visit_num}_{timestamp_str}.jpg"
                plate_crop_filename = f"{plate_number}_visit{visit_num}_{timestamp_str}_crop.jpg"
                
                snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)
                plate_crop_path = os.path.join(self.plate_crops_dir, plate_crop_filename)
                
                # Save images to disk
                if snapshot_img is not None and snapshot_img.size > 0:
                    cv2.imwrite(snapshot_path, snapshot_img)
                
                if plate_crop_img is not None and plate_crop_img.size > 0:
                    cv2.imwrite(plate_crop_path, plate_crop_img)
                
                # Insert visit history record
                cursor.execute("""
                    INSERT INTO visit_history 
                    (plate_number, visit_number, timestamp, snapshot_path, plate_crop_path, ocr_confidence) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (plate_number, visit_num, now, snapshot_path, plate_crop_path, confidence))
                
                self.conn.commit()
                logger.info(f"Recorded visit {visit_num} for plate {plate_number}")
                
                return visit_num
                
            except sqlite3.Error as e:
                self.conn.rollback()
                logger.error(f"Database error while recording visit for {plate_number}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while recording visit for {plate_number}: {e}")
                raise

    def get_vehicle_stats(self, plate_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve statistics and visit history for a specific vehicle.
        
        Args:
            plate_number (str): The license plate number to query.
            
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing vehicle stats and history, or None if not found.
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                
                # Get vehicle stats
                cursor.execute("SELECT * FROM vehicles WHERE plate_number = ?", (plate_number,))
                vehicle_row = cursor.fetchone()
                
                if not vehicle_row:
                    return None
                    
                # Get visit history
                cursor.execute("""
                    SELECT * FROM visit_history 
                    WHERE plate_number = ? 
                    ORDER BY timestamp DESC
                """, (plate_number,))
                history_rows = cursor.fetchall()
                
                return {
                    "plate_number": vehicle_row['plate_number'],
                    "total_visits": vehicle_row['total_visits'],
                    "first_seen": vehicle_row['first_seen'],
                    "last_seen": vehicle_row['last_seen'],
                    "history": [dict(row) for row in history_rows]
                }
            except sqlite3.Error as e:
                logger.error(f"Database error while retrieving stats for {plate_number}: {e}")
                return None

    def __del__(self):
        """Ensure database connection is closed upon object destruction."""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
