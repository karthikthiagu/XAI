import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt

imsize = 64



def regressor():

	sides = [3, 6]
	base = 'data/shapes/valid'
	for side in sides:
		if not os.path.isdir(base):
			os.mkdir(os.path.join(base, '{}'.format(side)))

	num_images = 1000
	min_cr = imsize / 5

	for side in sides:
		for i in range(num_images):
			xc, yc = np.random.randint(0, imsize, 2)
			while 1:
				max_cr = np.min(np.abs([xc - imsize, yc - imsize, xc, yc]))
				if max_cr <= min_cr:
					xc, yc = np.random.randint(0, imsize, 2)
					continue
				break

			cr = np.random.randint(min_cr, max_cr)
			thetap = np.random.randint(0, 360) * np.pi / 180
			xl, yl = xc + cr * np.sin(thetap), yc + cr * np.cos(thetap)
			image = np.zeros((imsize, imsize))
			for s in range(side):
				thetac = thetap + 2 * np.pi / side
				xr, yr = xc + cr * np.sin(thetac), yc + cr * np.cos(thetac)
				cv2.line(image, (int(xl), int(yl)), (int(xr), int(yr)), (255, 255, 255), 1)
				thetap = thetac
				xl, yl = xr, yr

			cv2.imwrite('data/shapes/valid/{}_{}.jpg'.format(side, i), image)


def classifier():

	def prepare(dat_type, seed, num_images):
		sides = [3, 4]
		base = 'data/shapes/classifier/{}'.format(dat_type)
		for side in sides:
			if not os.path.isdir(base):
				os.mkdir(os.path.join(base, '{}'.format(side)))

		np.random.seed(seed)
		min_cr = imsize / 5

		for side in sides:
			for i in range(num_images):
				xc, yc = np.random.randint(0, imsize, 2)
				while 1:
					max_cr = np.min(np.abs([xc - imsize, yc - imsize, xc, yc]))
					if max_cr <= min_cr:
						xc, yc = np.random.randint(0, imsize, 2)
						continue
					break

				cr = np.random.randint(min_cr, max_cr)
				thetap = np.random.randint(0, 360) * np.pi / 180
				xl, yl = xc + cr * np.sin(thetap), yc + cr * np.cos(thetap)
				image = np.zeros((imsize, imsize))
				for s in range(side):
					thetac = thetap + 2 * np.pi / side
					xr, yr = xc + cr * np.sin(thetac), yc + cr * np.cos(thetac)
					cv2.line(image, (int(xl), int(yl)), (int(xr), int(yr)), (255, 255, 255), 1)
					thetap = thetac
					xl, yl = xr, yr

				cv2.imwrite('data/shapes/classifier/{}/{}_{}.jpg'.format(dat_type, side, i), image)

	prepare('train', 1001, 100)
	prepare('valid', 1002, 50)

classifier()
