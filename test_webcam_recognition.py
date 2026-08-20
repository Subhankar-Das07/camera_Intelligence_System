import cv2
import logging
from pipelines.vehicle_recognition.pipeline import VehicleRecognitionPipeline
from pipelines.vehicle_recognition.database import VehicleDatabase

# Configure logging to see pipeline output during the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Vehicle Recognition Pipeline...")
    pipeline = VehicleRecognitionPipeline()
    # Initialize loads the models (YOLO, PaddleOCR, etc.) and database connections
    pipeline.initialize() 
    
    logger.info("Starting webcam (index 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Error: Could not open webcam.")
        return
        
    logger.info("Webcam started. Press 'q' to quit.")
    
    frame_idx = 0
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Failed to capture frame from webcam.")
                break
                
            # Process the frame through the pipeline
            annotated_frame, _ = pipeline.process_frame(
                frame, 
                frame_idx=frame_idx, 
                roi_polygon=None, 
                config={}
            )
            
            # Display the resulting frame
            cv2.imshow("Vehicle Recognition Webcam Test", annotated_frame)
            
            # Break the loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quitting test loop...")
                break
                
            frame_idx += 1
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    finally:
        # Cleanup OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Verify Database contents
        logger.info("Test finished. Querying database for summary...")
        db = VehicleDatabase()
        
        with db.lock:
            try:
                cursor = db.conn.cursor()
                cursor.execute("SELECT plate_number, total_visits, first_seen, last_seen FROM vehicles ORDER BY last_seen DESC")
                rows = cursor.fetchall()
                
                print("\n" + "="*70)
                print("DATABASE SUMMARY: Logged Vehicles")
                print("="*70)
                if not rows:
                    print("No vehicles logged in the database yet.")
                else:
                    for row in rows:
                        print(f"Plate: {row['plate_number']:<15} | Visits: {row['total_visits']:<5} | First: {row['first_seen']} | Last: {row['last_seen']}")
                print("="*70 + "\n")
            except Exception as e:
                logger.error(f"Error querying database: {e}")

if __name__ == "__main__":
    main()
