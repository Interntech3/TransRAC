Infer.py -> Cv2 and Transforms method; 

Infer_clean.py -> only Cv2 method

192.py -> logic of 192 frames taken in consideration with 64 frames processed in a batch of 3, aggregation of all 3 batches is the result. 

192_3.py -> logic of 192 frames with sliding window with adaptive aggregation approach (out of fixed batches, overlap with boundary correction, adaptive threshold, and sliding window , it gave best result)

Camera.py -> Integrates real time camera for repetition count, captures first 192 frames only.

Camera_2.py -> Integrates real time camera for repetition count, captures first 192 frames with 25 fps.
