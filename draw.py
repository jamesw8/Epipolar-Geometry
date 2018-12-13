import tkinter
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
import os
import random

from pprint import pprint

debug_mode = False
test_mode = False

image1_name = 'pic410.bmp'
image2_name = 'pic430.bmp'

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

def calculatePicture(normalize=True):
	global rimg1, img2, img2_canvas, width, epilines, epipoles, F_estimate
	# Reset epilines and epipoles drawn from last run
	if len(epilines) > 0:
		for epiline in epilines:
			w.delete(epiline[1])
		for epipole in epipoles:
			w.delete(epipole)
		epilines.clear()
		epipoles.clear()
	p_image, q_image = getPoints()
	p_image = [[i[0], i[1], 1] for i in p_image]
	q_image = [[i[0]-width, i[1], 1] for i in q_image]


	# TEST POINTS HERE
	if debug_mode:
		p_image = [(241.0, 417.0), (372.0, 525.0), (406.0, 316.0), (386.0, 262.0), (354.0, 97.0), (125.0, 244.0), (55.0, 578.0), (465.0, 60.0)]
		q_image = [(633.0, 462.0), (754.0, 574.0), (795.0, 359.0), (775.0, 306.0), (743.0, 144.0), (520.0, 284.0), (530.0, 587.0), (855.0, 105.0)]
		p_image = [[i[0], i[1], 1] for i in p_image]
		q_image = [[i[0]-width, i[1], 1] for i in q_image]

	print("List of points p:", p_image)
	print("List of points q:", q_image)

	F_estimate = calculateFundamentalMatrix(p_image, q_image, normalize=normalize)
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

def calculateFundamentalMatrix(p_image, q_image, normalize):
	p = q = None

	# Normalize the points for fundamental matrix calculation
	if normalize:
		print('\nP\n')
		pprint(p)
		print('\nQ\n')
		pprint(q)
		p, Tp = normalize_points(np.array(p_image))
		print('\nNew P\n')
		pprint(p)
		print('\nTp\n')
		pprint(Tp)
		q, Tq = normalize_points(np.array(q_image))
		print('\nNew Q\n')
		pprint(q)
		print('\nTq\n')
		pprint(Tq)
	else:
		p = p_image
		q = q_image
		Tp = np.identity(3)
		Tq = np.identity(3)

	A = np.empty((len(p_image), 9))

	assert len(p) == len(q) >= 8, "Need at least 8 point correspondences"
	print(len(p), len(q))

	for i in range(len(p)):
		xl = p[i][0]
		yl = p[i][1]
		xr = q[i][0]
		yr = q[i][1]

	
		A[i][0] = xl * xr 	# p[i][0] * q[i][0]
		A[i][1] = yl * xr 	# p[i][0] * q[i][1]
		A[i][2] = xr 		# p[i][0]
		A[i][3] = xl * yr 	# p[i][1] * q[i][0]
		A[i][4] = yl * yr 	# p[i][1] * q[i][1]
		A[i][5] = yr 		# p[i][1]
		A[i][6] = xl 		# q[i][0]
		A[i][7] = yl 		# p[i][1]
		A[i][8] = 1

	print('A rank', np.linalg.matrix_rank(A))
	assert np.linalg.matrix_rank(A) >= 8, 'A needs at least rank 8'
	print('A', A)
	U, s, Vt = np.linalg.svd(A)
	print(U.shape, Vt.shape, s.shape)
	print('s', len(s), s)

	x = Vt[-1].reshape((3, 3))

	print('F_hat rank', np.linalg.matrix_rank(x))

	# Test fundamental matrix equation
	# pr^T * F * pl = 0
	for i in range(len(p)):
		print(i, p[i], q[i], np.dot(np.transpose(q[i]), np.dot(x, p[i])))

	print('x', x.shape)
	pprint(x)

	skip_zero = False
	# Set smallest singular value to 0 to enforce rank of 2
	if not skip_zero:
		print('Setting smallest singular value to 0')
		# SVD of F_hat
		U, s, Vt = np.linalg.svd(x)

		D = np.diag([s[0], s[1], 0])

		F_estimate = np.dot(U, np.dot(D, Vt))

		print('F rank', np.linalg.matrix_rank(F_estimate))

		if normalize:
			# Denormalize F using transformation matrices from both sets of points
			F_estimate = np.dot(np.transpose(Tq), np.dot(F_estimate, Tp))

		# Test fundamental matrix equation
		# pr^T * F * pl = 0
		assert len(p_image) == len(q_image)
		for i in range(len(p_image)):
			print(i, p_image[i], q_image[i], np.dot(np.transpose(q_image[i]), np.dot(F_estimate, p_image[i])))

		print('F_estimate', F_estimate.shape)
		pprint(F_estimate)
		print('Dividing by bottom rightmost')
		F_estimate = F_estimate / F_estimate[2,2]
		print('New F_estimate')
		pprint(F_estimate)

		# SVD of F_estimate
		U, s, Vt = np.linalg.svd(F_estimate)

		D = np.diag(s)
		V = np.transpose(Vt)

		print('U', U.shape)
		pprint(U)
		print('D', D.shape)
		pprint(D)
		print('V', V.shape)
		pprint(V)

	return F_estimate

def normalize_points(points):
	x_total = y_total = 0
	x_count = y_count = 0

	for i in (points):
		x_total += i[0]
		y_total += i[1]
		x_count += 1
		y_count += 1
	
	x_mean = x_total / x_count
	y_mean = y_total / y_count

	S = scaling_factor = (2**0.5) / np.std(np.concatenate([points[:,0], points[:,1]]))
	print('S', S)
	# Translation matrix to normalize points
	T = np.array([
		[S, 0, -S*x_mean],
		[0, S, -S*y_mean],
		[0, 0, 1]
	])
	new_points = []
	for i in range(len(points)):					
		new_points.append(np.matmul(T, points[i]))
	# print('Normalized', new_points)
	return new_points, T
# Get epipoles
def calculateEpipoles(F):
	U, s, Vt = np.linalg.svd(F)
	e1 = Vt[-1]
	U, s, Vt = np.linalg.svd(np.transpose(F))
	e2 = Vt[-1]
	return e1, e2
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
	# e_x = e[0] / e[2]
	# e_y = e[1] / e[2]
	# print('Epipole', e_x, e_y)
	l = np.dot(F, point)

	print('Epipolar line', l)

	# if e_x < 0 or e_x > width or e_y < 0 or e_y > height:
	# 	w.itemconfigure(coord, text=w.itemcget(coord, 'text')+' Out of bounds')
	# 	return
	y = lambda x: (l[2] + l[0]*x)/(-l[1]) # -(l[0]*x + l[2]) / l[1]
	x0 = 0
	y0 = y(x0)
	x1 = width - 1
	y1 = y(x1)
	line = w.create_line(x0 + add, y0, x1 + add, y1, fill=color, width=2) # , arrow=tkinter.LAST)
	epilines.append(line)
	# line = w.create_line(l[0], l[1], point[0], point[1], fill='pink', width=2) # , arrow=tkinter.LAST)
# Draw test point epipolar lines
def createTestEpipolarLines():
	global w, test_epilines
	assert F_estimate != [], "Need fundamental matrix first"

	for epiline in test_epilines:
		w.delete(epiline)
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
	global test_mode
	if not test_mode:
		testButton.config(state="normal", text="Add control points")
	else:
		testButton.config(state="normal", text="Add test points")
	test_mode = not test_mode
	print("Test Mode: " + str(test_mode))
# Toggle visibility of test points and arrows
def toggleTestPoints():
	global w, test_new, test_original, test_arrows, test_correct, test_hidden, test_epilines
	if test_hidden:
		for i in test_new + test_original + test_arrows + test_correct + test_epilines:
			w.itemconfigure(i, state="normal")
		test_hidden = False
	else:
		for i in test_new + test_original + test_arrows + test_correct + test_epilines:
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
		# oval = w.create_oval(x_min-9, y_min-9, x_min+9, y_min+9, width=0, fill="#ff0000",activefill="#ff0000",disabledfill="#ff0000")
		# text = w.create_text(x_min+9, y_min+9, text=str(min_d))
		oval = w.create_oval(x_min-9+width, y_min-9, x_min+9+width, y_min+9, width=0, fill="#ff0000",activefill="#ff0000",disabledfill="#ff0000")
		text = w.create_text(x_min+9+width, y_min+9, text=str(min_d))
		test_correct.append((oval, text))

'''
	line_vec = vector(start, end)
	pnt_vec = vector(start, pnt)
	line_len = length(line_vec)
	line_unitvec = unit(line_vec)
	pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
	t = dot(line_unitvec, pnt_vec_scaled)    
	if t < 0.0:
		t = 0.0
	elif t > 1.0:
		t = 1.0
	nearest = scale(line_vec, t)
	dist = distance(nearest, pnt_vec)
	nearest = add(nearest, start)
	return (dist, nearest)
	'''
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
def main():
	global w, width, height, new, original, arrows, coord, rimg1, img2, img2_canvas, calculateButton, epilines, epipoles, hidden, test_new, test_original, test_arrows, test_hidden, testButton, test_epilines, test_correct, F_estimate
	# Initialize window and canvas
	top = tkinter.Tk()
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

	resize = False

	if resize:
		rimg1 = rimg1.resize((640, 480))
		rimg2 = rimg2.resize((640, 480))
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
	main()
