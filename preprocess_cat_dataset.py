## Modified version of https://github.com/microe/angora-blue/blob/master/cascade_training/describe.py by Erik Hovland
##### I modified it to work with Python 3, changed the paths and made it output a folder for images bigger than 64x64 and a folder for images bigger than 128x128

import cv2
import glob
import math
import sys

def rotateCoords(coords, center, angleRadians):
	# Positive y is down so reverse the angle, too.
	angleRadians = -angleRadians
	xs, ys = coords[::2], coords[1::2]
	newCoords = []
	n = min(len(xs), len(ys))
	i = 0
	centerX = center[0]
	centerY = center[1]
	cosAngle = math.cos(angleRadians)
	sinAngle = math.sin(angleRadians)
	while i < n:
		xOffset = xs[i] - centerX
		yOffset = ys[i] - centerY
		newX = xOffset * cosAngle - yOffset * sinAngle + centerX
		newY = xOffset * sinAngle + yOffset * cosAngle + centerY
		newCoords += [newX, newY]
		i += 1
	return newCoords

def preprocessCatFace(coords, image):

	leftEyeX, leftEyeY = coords[0], coords[1]
	rightEyeX, rightEyeY = coords[2], coords[3]
	mouthX = coords[4]
	if leftEyeX > rightEyeX and leftEyeY < rightEyeY and \
			mouthX > rightEyeX:
		# The "right eye" is in the second quadrant of the face,
		# while the "left eye" is in the fourth quadrant (from the
		# viewer's perspective.) Swap the eyes' labels in order to
		# simplify the rotation logic.
		leftEyeX, rightEyeX = rightEyeX, leftEyeX
		leftEyeY, rightEyeY = rightEyeY, leftEyeY

	eyesCenter = (0.5 * (leftEyeX + rightEyeX),
				  0.5 * (leftEyeY + rightEyeY))

	eyesDeltaX = rightEyeX - leftEyeX
	eyesDeltaY = rightEyeY - leftEyeY
	eyesAngleRadians = math.atan2(eyesDeltaY, eyesDeltaX)
	eyesAngleDegrees = eyesAngleRadians * 180.0 / math.pi

	# Straighten the image and fill in gray for blank borders.
	rotation = cv2.getRotationMatrix2D(
			eyesCenter, eyesAngleDegrees, 1.0)
	imageSize = image.shape[1::-1]
	straight = cv2.warpAffine(image, rotation, imageSize,
							  borderValue=(128, 128, 128))

	# Straighten the coordinates of the features.
	newCoords = rotateCoords(
			coords, eyesCenter, eyesAngleRadians)

	# Make the face as wide as the space between the ear bases.
	w = abs(newCoords[16] - newCoords[6])
	# Make the face square.
	h = w
	# Put the center point between the eyes at (0.5, 0.4) in
	# proportion to the entire face.
	minX = eyesCenter[0] - w/2
	if minX < 0:
		w += minX
		minX = 0
	minY = eyesCenter[1] - h*2/5
	if minY < 0:
		h += minY
		minY = 0

	# Crop the face.
	crop = straight[int(minY):int(minY+h), int(minX):int(minX+w)]
	# Return the crop.
	return crop

def describePositive():
	output = open('log.txt', 'w')
	for imagePath in glob.glob('cat_dataset/*.jpg'):
		# Open the '.cat' annotation file associated with this
		# image.
		input = open('%s.cat' % imagePath, 'r')
		# Read the coordinates of the cat features from the
		# file. Discard the first number, which is the number
		# of features.
		coords = [int(i) for i in input.readline().split()[1:]]
		# Read the image.
		image = cv2.imread(imagePath)
		# Straighten and crop the cat face.
		crop = preprocessCatFace(coords, image)
		if crop is None:
			print >> sys.stderr, \
					'Failed to preprocess image at %s.' % \
					imagePath
			continue
		# Save the crop to folders based on size
		h, w, colors = crop.shape
		if min(h,w) >= 64:
			Path1 = imagePath.replace("cat_dataset","cats_bigger_than_64x64")
			cv2.imwrite(Path1, crop)
		if min(h,w) >= 128:
			Path2 = imagePath.replace("cat_dataset","cats_bigger_than_128x128")
			cv2.imwrite(Path2, crop)
		# Append the cropped face and its bounds to the
		# positive description.
		#h, w = crop.shape[:2]
		#print (cropPath, 1, 0, 0, w, h, file=output)


def main():
	describePositive()

if __name__ == '__main__':
	main()
