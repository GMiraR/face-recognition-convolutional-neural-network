import face_recognition
import os
import cv2

# Directories
KNOWN_FACES = 'known_faces'
UNKNOWN_FACES = 'unknown_faces'

# To match the images '0.6 to avoid false positives'
TOLERANCE = 0.6
# Drawn a square around the head
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

# Convolutional Neural Networks - to analyze images
MODEL = 'cnn'

print('Loading known faces...')

known_faces = []
# kind of labels
known_names = []

for name in os.listdir(KNOWN_FACES):
	for filename in os.listdir(f'KNOWN_FACES/{name}'):
		#loading the image
		image = face_recognition.load_image_file(f"{KNOWN_FACES}/{name}/{filename}")
		#encode the image (codificar) encode all the faces that it finds
		#using the first index to use as identity for the recognition
		encoding = face_recognition.face_encodings(image)[0]
		known_faces.append(encoding)
		known_names.append(name)
		# we don't care about the location/coordinate for the known faces
		# we just care about it's identity (it's going to be used for any comparison)


print('Processing unkown faces...')

#no need to iterate over the directory because we aren't looking for an identity
for filename in os.listdir(UNKNOWN_FACES):
	print(filename)
	image = face_recognition.load_image_file(f'{UNKNOWN_FACES}/{filename}')
	# face detection - find all the coordinates for faces in the unkown dir
	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = face_recognition.face_encodings(image, locations)
	#convert image for visualization purposes
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	#looking for matches
	for face_encoding, face_location in zip(encodings, locations):
		#compare the current encoding to every single known face
		#then it returns a list of booleans (true or false) 
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

		match = None

		#if at least we got 1 match (1 True)
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match found: {match}")

			#draw a rectangle
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])

			color = [0, 255, 0]

			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
			
			# little rectangle with label
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2] + 22 )#22 up
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match, face_location[3] + 10, face_location[0] + 15, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS))
	cv2.imshow(filename, image)
	cv2.waitKey(10000)

	#cv2.destroyWindow(filename)
