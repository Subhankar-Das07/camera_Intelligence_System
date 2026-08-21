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
        
        # Verify Database contents (Redis)
        logger.info("Test finished. Querying Redis for summary...")
        db = VehicleDatabase()
        try:
            plates = [p.decode() if isinstance(p, bytes) else str(p)
                      for p in db.r.smembers("vr:plates")]
            print("\n" + "=" * 70)
            print("DATABASE SUMMARY: Logged Vehicles (Redis)")
            print("=" * 70)
            if not plates:
                print("No vehicles logged in the database yet.")
            else:
                for plate in sorted(plates):
                    stats = db.get_vehicle_stats(plate)
                    if not stats:
                        continue
                    print(
                        f"Plate: {stats['plate_number']:<15} | "
                        f"Visits: {stats['total_visits']:<5} | "
                        f"First: {stats['first_seen']} | Last: {stats['last_seen']}"
                    )
            print("=" * 70 + "\n")
        except Exception as e:
            logger.error(f"Error querying database: {e}")

if __name__ == "__main__":
    main()
