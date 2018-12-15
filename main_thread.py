import argparse
import threading
from subprocess import call


# Script calls
def call_draw(algorithm):
	try:
		call(["python", "draw.py", "-a", algorithm])
	except:
		call(["python3", "draw.py", "-a", algorithm])

if __name__ == '__main__':
	thread1 = threading.Thread(target=call_draw, args=('7point',))
	thread2 = threading.Thread(target=call_draw, args=('8point',))
	thread3 = threading.Thread(target=call_draw, args=('RANSAC',))
	thread4 = threading.Thread(target=call_draw, args=('LMEDS',))
	thread1.start()
	thread2.start()
	thread3.start()
	thread4.start()

