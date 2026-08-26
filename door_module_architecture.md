# Vision Watch Door Module Architecture

This document explains the technical approach, algorithms, and flow used for the Vision Watch door open and close tracking module.

## The Core Idea: "Structural Comparator"
Instead of looking for general "motion" (which gets confused when a person walks past a closed door), the system now uses a **Structural Comparator**. It memorizes exactly what the closed door looks like, and only triggers when the *physical structure* of that specific area changes significantly.

The entire logic runs on **Python** using two powerful, lightweight libraries: **OpenCV (cv2)** and **NumPy**. By keeping it strict to classic Computer Vision operations and a Python State Machine, the module runs incredibly fast, uses almost zero CPU/GPU compared to AI models, and provides predictable accuracy based on deterministic rules.

---

## Step 1: The Learning Phase (Median Filtering)
When you draw the polygon (ROI) over the door and click "Start", the system spends the first 20 frames simply "looking" at the door.

* **Tech Used:** `NumPy`
* **Algorithm:** **Temporal Median Filtering**
* **How it works:** We take the pixels from the 20 frames and stack them into a 3D matrix using `numpy.stack()`. We then use `numpy.median()` to find the exact "middle" color value for every single pixel over time. If a person walked past the door in frame 5, they are an "outlier." The median completely ignores them, resulting in a perfectly clean, mathematical background image of the closed door.

---

## Step 2: Live Comparison (Absolute Difference & Morphology)
As the video plays, the system takes the pixels inside your drawn polygon and compares them against the "Closed Door" picture it memorized.

* **Tech Used:** `OpenCV` (cv2)
* **Algorithm:** **Background Subtraction & Morphological Operations**
* **How it works:** 
  1. **Absolute Difference:** We use `cv2.absdiff()` to subtract the current frame from our perfect "Median" closed door. Any pixels that changed light up.
  2. **Binary Thresholding:** We use `cv2.threshold()` to convert the image to pure black-and-white. Black means "no change", white means "changed."
  3. **Morphological Opening & Dilation:** Camera sensors have noise (tiny flickering pixels). We use `cv2.morphologyEx(..., cv2.MORPH_OPEN)` to mathematically erase isolated specks of noise, and `cv2.dilate()` to group the real, dense changes together. Finally, `cv2.countNonZero()` gives us the exact count of "changed" pixels to calculate the **Activity Score**.

---

## Step 3: The State Machine (Finite State Machine)
To prevent the count from flickering wildly if the door is halfway open, the system uses strict rules for counting.

* **Tech Used:** Pure **Python** logic (Dictionaries and `if/elif` statements)
* **Architecture:** **Deterministic Finite State Machine (FSM) with Temporal Debouncing**
* **How it works:** The system operates in predefined "States" (`INITIALIZING`, `CLOSED`, `OPEN`). It is literally a Python dictionary tracking the current state, and the rules dictate that it cannot transition to a new state unless the Activity Score threshold is breached for `N` consecutive frames (tracked by a simple integer counter variable called temporal debouncing). This guarantees we never get "double counts" from a slightly vibrating door.
  * **To Open:** The Activity Score must jump above **15%**, and it must *stay* above 15% for at least 4 consecutive frames. Once it hits this, the state changes to **OPEN** and the open count goes up by 1.
  * **To Close:** The Activity Score must drop below **5%** (meaning the door looks exactly like the closed picture again) and stay there for 4 consecutive frames. The state changes back to **CLOSED** and the close count goes up.

---

## Step 4: Report Generation
* Throughout the video, this live state (the counts) is held in the pipeline's memory. 
* When you hit **Stop**, the backend explicitly pulls the final `open_count` and `close_count` from the live state, packages it into a strict `vision_watch` JSON format, and saves it to the Redis database.
* The frontend then fetches this specific report and displays the final door counts on the UI, completely isolated from any face recognition data.
