# NYC Then and Now
We can find the differences in old and new pictures of NYC given that we can combine these to create stereo pairs. We can then derive the epipolar geometry without any knowledge of intrinsic parameters of the cameras. Then, we can draw epipolar lines so that we can search for matches between the pair of images.

## Instructions
1. Create a virtual environment using `virtualenv .venv` and activate. On Unix `source .venv/bin/activate` and on Windows `.venv/scripts/activate`.
2. Install libraries needed including OpenCV by entering `pip{3} install -r requirements.txt`.
* To run individual algorithms, run `python{3} draw.py -a {7point,8point,RANSAC,LMEDS}`. For example, to use the 8 point algorithm `python{3} draw.py -a 8point`.
* To run four instances using all algorithms, run `python{3} main_thread.py`.

Enjoy!
