import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt

imsize = 512
font = cv2.FONT_HERSHEY_SIMPLEX


def writeImages(dat_type, num_images, seed):

	np.random.seed(seed)
	denoms = [1, 2, 5, 10, 20]
	rads = [20, 25, 30, 35, 40]
	num_denoms = len(denoms)
	base = 'data/denom/{}'.format(dat_type)

	single_counter = 0
	for i in range(num_images):
		objects = []
		denoms_count = np.random.randint(0, 3, num_denoms)
		if i % num_denoms == 0:
			denoms_count = np.zeros((num_denoms))
			denoms_count[single_counter] = 1
			single_counter += 1
			if single_counter == 5:
				single_counter = 0
		value = np.dot(denoms, denoms_count)
		denoms_zip = zip(denoms_count, denoms, rads)
		image = np.zeros((imsize, imsize))
		print(i)

		for item in denoms_zip: 
			denom_count, denom, denom_rad = item
			while denom_count > 0:
				x, y = 0, 0
				while np.min([x, y, imsize - x, imsize - y]) <= denom_rad:
					#print('x = {}, y = {}, denom_rad = {}, denom_count = {}, denom = {}'.format(x, y, denom_rad, denom_count, denom))
					x, y = np.random.randint(0, imsize, 2)
					for xe, ye, re in objects:
						if np.sqrt((xe - x) ** 2 + (ye - y) ** 2) <= re + denom_rad:
							x, y = 0, 0
							break
				#input()
				cv2.circle(image, (x, y), denom_rad, (255, 255, 255))
				cv2.putText(image, '{}'.format(denom),(x - 10, y + 10), font, 1, (255,255,255), 1, cv2.LINE_AA)
				objects.append((x, y, denom_rad))
				denom_count -= 1
		image = cv2.resize(image, (256, 256))
		cv2.imwrite('data/denom/{}/{}_{}.jpg'.format(dat_type, value, i), image)

writeImages('train', 1000, 1001)
writeImages('valid', 500, 1002)
