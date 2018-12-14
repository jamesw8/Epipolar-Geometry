import cv2
import numpy as np

from pprint import pprint

def calculate_8point_fundamental_matrix(p_image, q_image, normalize=True):
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

	return F_estimate, [1]*len(p_image)

def calculate_7point_fundamental_matrix(p_image, q_image):
	F, mask = cv2.findFundamentalMat(p_image, q_image, cv2.FM_7POINT)
	return F, mask

# TODO: Add extra params
def calculate_RANSAC_fundamental_matrix(p_image, q_image):
	F, mask = cv2.findFundamentalMat(p_image, q_image, cv2.FM_RANSAC)
	return F, mask

def calculate_LMEDS_fundamental_matrix(p_image, q_image):
	F, mask = cv2.findFundamentalMat(p_image, q_image, cv2.FM_LMEDS)
	return F, mask

# Normalize points for 8point algorithm
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
	print('Scaling factor', S)
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