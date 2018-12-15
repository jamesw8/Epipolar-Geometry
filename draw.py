'''
James Wong 1903
'''

import tkinter
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
import os
import random
import argparse

# Import fundamental matrix algorithms
from fundamental import calculate_8point_fundamental_matrix, \
						calculate_7point_fundamental_matrix, \
						calculate_RANSAC_fundamental_matrix, \
						calculate_LMEDS_fundamental_matrix,  \
						normalize_points,					 \
						calculateEpipoles

from pprint import pprint

# Parse arguments
def get_args():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-a', '--algorithm', choices=['7point', '8point', 'RANSAC', 'LMEDS'],
	                    help='algorithm used to calculate fundamental matrix', required=True)
	
	args = parser.parse_args()
	return args

algorithm_name = get_args().algorithm

# Get algorithm function from argument
def get_algorithm_from_args(algorithm_name):
	if algorithm_name == '7point':
		return calculate_7point_fundamental_matrix
	elif algorithm_name == '8point':
		return calculate_8point_fundamental_matrix
	elif algorithm_name == 'RANSAC':
		return calculate_RANSAC_fundamental_matrix
	elif algorithm_name == 'LMEDS':
		return calculate_LMEDS_fundamental_matrix
	else:
		assert False, "Can't find specified algorithm " + args.algorithm

algorithm = get_algorithm_from_args(algorithm_name)
debug_mode = True
debug_p_image = [(254.0, 140.0), (393.0, 147.0), (252.0, 173.0), (397.0, 177.0), (369.0, 94.0), (400.0, 65.0), (334.0, 61.0), (267.0, 229.0), (160.0, 344.0), (381.0, 233.0), (511.0, 198.0)]
debug_q_image = [(973.0, 209.0), (1107.0, 198.0), (970.0, 236.0), (1113.0, 231.0), (1157.0, 162.0), (1182.0, 138.0), (1122.0, 143.0), (983.0, 287.0), (730.0, 400.0), (1094.0, 286.0), (1220.0, 231.0)]

preset_test_points_mode = True
preset_test_p = [(254.0, 140.0), (393.0, 147.0), (252.0, 173.0), (397.0, 177.0), (369.0, 94.0), (400.0, 65.0), (334.0, 61.0), (267.0, 229.0), (160.0, 344.0), (381.0, 233.0), (511.0, 198.0)]
preset_test_q = [(973.0, 209.0), (1107.0, 198.0), (970.0, 236.0), (1113.0, 231.0), (1157.0, 162.0), (1182.0, 138.0), (1122.0, 143.0), (983.0, 287.0), (730.0, 400.0), (1094.0, 286.0), (1220.0, 231.0)]

test_mode = False
normalize = True
resize = True
new_image_size = ((640, 480))

image1_name = 'washington_park_old.jpg'
image2_name = 'washington_park_new.jpg'

# Listener callbacks
def listenClick(event):
	global w, current, new, calculateButton, test_new
	print('Clicking', event.x, event.y)
	if not test_mode:
		for pt in new:
			point = w.coords(pt)
			if (event.x >= point[0] and event.x <= point[2]) and (event.y >= point[1] and event.y <= point[3]):
				print('Exists', w.type(pt))
				current = pt
				return
		print('Creating point')
		createPoint(event)
		if len(new) >0:
			calculateButton.config(state='normal', text='Calculate')
	else:
		for pt in test_new:
			point = w.coords(pt)
			if (event.x >= point[0] and event.x <= point[2]) and (event.y >= point[1] and event.y <= point[3]):
				print('Exists', w.type(pt))
				current = pt
				return
		print('Creating point')
		createPoint(event)
		if len(test_new) >0:
			calculateButton.config(state='normal', text='Calculate')
def listenDrag(event):
	global w, current, new, original, arrows, test_new, test_original, test_arrows
	print('Dragging', event.x, event.y)
	print(current != None)
	if current != None:
		print('Dragging it!', event.x, event.y)
		movePoint(event)
		if not test_mode:
			for pt in range(len(new)):
				if current == new[pt]:
					new_coords = getActualCoords(new[pt])
					orig_coords = getActualCoords(original[pt])
					old_coords = w.coords(arrows[pt])
					w.coords(arrows[pt], old_coords[0], old_coords[1], new_coords[0], new_coords[1])
		else:
			for pt in range(len(test_new)):
				if current == test_new[pt]:
					new_coords = getActualCoords(test_new[pt])
					orig_coords = getActualCoords(test_original[pt])
					old_coords = w.coords(test_arrows[pt])
					w.coords(test_arrows[pt], old_coords[0], old_coords[1], new_coords[0], new_coords[1])
def listenRelease(event):
	global current, img2
	print('Releasing', event.x, event.y)
	current = None
def listenHover(event):
	updateMouseCoord(event)

def calculatePicture(normalize=normalize):
	global rimg1, img2, img2_canvas, width, epilines, epipoles, F_estimate, debug_p_image, debug_q_image
	# Reset epilines and epipoles drawn from last run
	if len(epilines) > 0:
		for epiline in epilines:
			w.delete(epiline)
		for epipole in epipoles:
			w.delete(epipole)
		epilines.clear()
		epipoles.clear()
	p_image, q_image = getPoints()
	p_image = [[i[0], i[1], 1] for i in p_image]
	q_image = [[i[0]-width, i[1], 1] for i in q_image]


	# TEST POINTS HERE
	if debug_mode:
		p_image.extend(debug_p_image)
		q_image.extend(debug_q_image)

		p_image = [[i[0], i[1], 1] for i in p_image]
		q_image = [[i[0]-width, i[1], 1] for i in q_image]

	print("List of points p:", p_image)
	print("List of points q:", q_image)
	p_image = np.float64(p_image)
	q_image = np.float64(q_image)

	print('Using', algorithm)
	F_estimate, mask = algorithm(p_image, q_image)#calculateFundamentalMatrix(p_image, q_image, normalize=normalize)
	print(F_estimate)
	# Draw epipoles (2 methods)
		# THIS WORKS
		# e1 = U[:,2]
		# e2 = V[:,2]
		# drawEpipole(e1)
		# drawEpipole(e2)
		# print('e1', np.transpose(e1))
		# print('e2', np.transpose(e2))

		# print('Getting epipoles using only V')
		# THIS WORKS TOO
	e1, e2 = calculateEpipoles(F_estimate)
	print('Epipoles', e1, e2)
	drawEpipole(e1, 0)
	drawEpipole(e2, width)

	random_color = lambda: random.randint(0,255)
	for i in range(len(p_image)):
		color = '#%02X%02X%02X' % (random_color(),random_color(),random_color())

		createEpipolarLine(F_estimate, p_image[i], width, color)
		createEpipolarLine(np.transpose(F_estimate), q_image[i], 0, color)


# Draw epipole
def drawEpipole(e, add):
	global w, width, height, new, coord, epipoles
	e_x = e[0] / e[2]
	e_y = e[1] / e[2]
	print('Drawing epipole at ', e_x, ',', e_y)
	epipole = w.create_oval(e_x - 9 + add, e_y - 9, e_x + 9 + add, e_y + 9, width=0, fill="orange")
	epipoles.append(epipole)
# Draw epipolar line
def createEpipolarLine(F, point, add, color):
	global w, width, height, new, coord, epilines

	l = np.dot(F, point)

	print('Epipolar line', l)

	y = lambda x: (l[2] + l[0]*x)/(-l[1]) 
	x0 = 0
	y0 = y(x0)
	x1 = width - 1
	y1 = y(x1)
	line = w.create_line(x0 + add, y0, x1 + add, y1, fill=color, width=2)
	epilines.append(line)
	
# Draw test point epipolar lines
def createTestEpipolarLines():
	global w, test_epilines

	assert F_estimate != [], "Need fundamental matrix first"

	for epiline in test_epilines:
		w.delete(epiline[1])
	test_epilines.clear()

	for single_test_point in test_original:
		point = getActualCoords(single_test_point)
		l = np.dot(F_estimate, (point[0], point[1], 1))

		y = lambda x: (l[2] + l[0]*x)/(-l[1])
		x0 = 0
		y0 = y(x0)
		x1 = width - 1
		y1 = y(x1)
		line = w.create_line(x0 + width, y0, x1 + width, y1, fill="#0000ff", width=2)
		test_epilines.append((l, line))
	calculateError()

# Toggle visibility of points and arrows
def togglePoints():
	global w, new, original, arrows, hidden
	if hidden:
		for i in new + original + arrows:
			w.itemconfigure(i, state='normal')
		hidden = False
	else:
		for i in new + original + arrows:
			w.itemconfigure(i, state='hidden')
		hidden = True
# Print points to reuse
def printPoints():
	original, new = getPoints()
	print('Original', original)
	print('New', new)
# Add test points to test fundamental matrix
def testPoints():
	global test_mode, preset_test_points_mode
	if not test_mode:
		if preset_test_points_mode:
			test_original = preset_test_p
			test_new = preset_test_q
			for index in range(len(test_original)):
				x1 = test_original[index][0]
				y1 = test_original[index][1]
				x2 = test_new[index][0]
				y2 = test_new[index][1]
				#print(x1, y1, x2,y2)
				arrow = w.create_line(x1, y1, x2, y2, width=2, arrow=tkinter.LAST)
				test_arrows.append(arrow)
			createTestEpipolarLines()
			testButton.config(state="normal", text="Add control points")
	else:
		testButton.config(state="normal", text="Add test points")
	test_mode = not test_mode
	print("Test Mode: " + str(test_mode))
# Toggle visibility of test points and arrows
def toggleTestPoints():
	global w, test_new, test_original, test_arrows, test_correct, test_hidden, test_epilines
	if test_hidden:
		for i in test_new + test_original + test_arrows + test_correct + [epiline[1] for epiline in test_epilines]:
			w.itemconfigure(i, state="normal")
		test_hidden = False
	else:
		for i in test_new + test_original + test_arrows + test_correct + [epiline[1] for epiline in test_epilines]:
			w.itemconfigure(i, state="hidden")
		test_hidden = True
# Calculate error in test point epilines
def calculateError():
	global w, test_new, test_epilines, test_correct
	assert(len(test_epilines) > 0), "there are no test epilines"
	for pair in test_correct:
		# Oval pair[0]
		# Text pair[1]
		w.delete(pair[0])
		w.delete(pair[1])
	test_correct.clear()

	d = lambda x0, y0, a, b, c: np.abs((a*x0) + (b*y0) + c) / np.sqrt(a**2 + b**2)
	x = lambda x0, y0, a, b, c: (((b**2) * x0) - (a*c) - (a*b*y0)) / (a**2 + b**2)
	y = lambda x_ans, a, b, c: ((a*x_ans) + c) / (-1*b)
	print('Test epilines', test_epilines)
	for point, epiline in zip(test_new, test_epilines):
		print(getActualCoords(point), epiline[0])
		p_coords = getActualCoords(point)
		line = epiline[0]
		min_d = d(p_coords[0]-width, p_coords[1], line[0], line[1], line[2])
		print('Min dist', min_d)
		x_min = x(p_coords[0]-width, p_coords[1], line[0], line[1], line[2])
		y_min = y(x_min, line[0], line[1], line[2])
		print('Correct point', x_min, y_min)
		oval = w.create_oval(x_min-9+width, y_min-9, x_min+9+width, y_min+9, width=0, fill="#ff0000",activefill="#ff0000",disabledfill="#ff0000")
		text = w.create_text(x_min+9+width, y_min+9, text=str(min_d))
		test_correct.append((oval, text))

# Clears test points
def clearTestPoints():
	global test_new, test_original, test_arrows, test_correct, test_epilines

	assert(len(test_original) > 0), "there are no test points"

	for point in test_new:
		w.delete(point)
	for point in test_original:
		w.delete(point)
	for arrow in test_arrows:
		w.delete(arrow)
	for oval_text in test_correct:
		w.delete(oval_text[0])
		w.delete(oval_text[1])
	for epiline in test_epilines:
		w.delete(epiline[1])
	test_new = []
	test_original = []
	test_arrows = []
	test_correct = []
	test_epilines = []

# Create points
def createPoint(event):
	global w, width, height, new, coord
	if event.x < 0 or event.x > width or event.y < 0 or event.y > height:
		w.itemconfigure(coord, text=w.itemcget(coord, 'text')+' Out of bounds')
		return
	x = event.x
	y = event.y
	if not test_mode:
		original.append(w.create_oval(x-9, y-9, x+9, y+9, width=0, fill="#ff0000",activefill="#ff0000",disabledfill="#ff0000"))
		new.append(w.create_oval(x-9, y-9, x+9, y+9, width=0, fill="#00ff00"))
		arrow = w.create_line(x, y, x, y, width=2, arrow=tkinter.LAST)
		arrows.append(arrow)
	else:
		test_original.append(w.create_oval(x-9, y-9, x+9, y+9, width=0, fill="#ff00ff",activefill="#ff00ff",disabledfill="#ff00ff"))
		test_new.append(w.create_oval(x-9, y-9, x+9, y+9, width=0, fill="#0000ff"))
		arrow = w.create_line(x, y, x, y, width=2, arrow=tkinter.LAST)
		test_arrows.append(arrow)
# Move point
def movePoint(event):
	global width, height
	if event.x < 0:
		x = 0
	elif event.x > width*2:
		x = width
	else:
		x = event.x
	if event.y < 0:
		y = 0
	elif event.y > height:
		y = height
	else:
		y = event.y
		error_msg = ' Out of bounds' if x != event.x or y != event.y else ''
		w.coords(current, x-9, y-9, x+9, y+9)
		w.itemconfigure(coord, text='%d, %d'%(event.x, event.y) + error_msg)
# Get points
def getPoints():
	global original, new
	return list(map(getActualCoords, original)), list(map(getActualCoords, new))
# Get picture
def getPicture(pic):
	return np.asarray(pic)
def arrayToPicture(arr):
	return Image.fromarray(np.uint8(arr))
def getActualCoords(point):
	# Get coords from a circle
	coords = w.coords(point)
	return coords[0]+9, coords[1]+9
def updateMouseCoord(event):
	global w, coord
	w.itemconfigure(coord, fill='white', text='%d, %d'%(event.x, event.y))
def createWindow():
	global w, width, height, new, original, arrows, coord, rimg1, img2, img2_canvas, calculateButton, \
	epilines, epipoles, hidden, test_new, test_original, test_arrows, test_hidden, testButton, test_epilines, test_correct, \
	F_estimate, resize, new_image_size, algorithm_name
	# Initialize window and canvas
	top = tkinter.Tk()
	top.title(algorithm_name)
	w = tkinter.Canvas(top, bd=-2)
	w.grid(row=0, column=0)
	# Event Listeners
	w.bind('<Button-1>', listenClick)
	w.bind('<B1-Motion>', listenDrag)
	w.bind('<ButtonRelease-1>', listenRelease)
	w.bind('<Motion>', listenHover)

	# Open Image
	rimg1 = Image.open(image1_name)
	rimg2 = Image.open(image2_name)

	if resize:
		rimg1 = rimg1.resize(new_image_size)
		rimg2 = rimg2.resize(new_image_size)
	[width, height] = rimg1.size

	# Set window to twice width to fit two pictures
	w.config(width=width*2, height=height)
	img1 = ImageTk.PhotoImage(rimg1)

	# Create images
	w.create_image(0, 0, image=img1, anchor="nw")
	w.create_line(width, 0, width, height)
	img2 = ImageTk.PhotoImage(rimg2)
	img2_canvas = w.create_image(width,0, image=img2, anchor="nw")
	f = tkinter.Frame(height=50)
	calculateButton = tkinter.Button(f,text="Need to add points", state='normal', command=calculatePicture)
	hideButton = tkinter.Button(f,text="Toggle control points", state='normal', command=togglePoints)
	printButton = tkinter.Button(f, text="Export points", state='normal', command=printPoints)
	testButton = tkinter.Button(f, text="Add test points", state="normal", command=testPoints)
	toggleTestButton = tkinter.Button(f, text="Toggle test points", state="normal", command=toggleTestPoints)
	testEpilines = tkinter.Button(f, text="Calculate test point epilines", state="normal", command=createTestEpipolarLines)
	findError = tkinter.Button(f, text="Calculate error", state="normal", command=calculateError)
	resetTestPoints = tkinter.Button(f, text="Clear test points", state="normal", command=clearTestPoints)
	hidden = False
	test_hidden = False

	# Store points and lines
	current = None
	new = []
	original = []
	arrows = []
	epilines = []
	epipoles = []
	test_new = []
	test_original = []
	test_correct = []
	test_arrows = []
	test_epilines = []

	#F_estimate intialize
	F_estimate = []

	# Coordinate indicator
	coord = w.create_text(10, height)
	w.itemconfigure(coord, text='0 0', anchor="sw")
	top.geometry('{}x{}'.format(2*width, height+50))
	calculateButton.grid(row=0, column=0)
	hideButton.grid(row=0, column=1)
	printButton.grid(row=0, column=2)
	testButton.grid(row=0, column=3)
	toggleTestButton.grid(row=0, column=4)
	testEpilines.grid(row=0, column=5)
	findError.grid(row=0, column=6)
	resetTestPoints.grid(row=0, column=7)
	w.grid(row=1)
	f.grid(row=0)
	top.mainloop()

if __name__ == '__main__':
	createWindow()
