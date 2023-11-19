import sys
sys.path.append("../")
from core import detectWithSearchName_v2, magnifier, closenessHelper, isObjectPresent
import cv2 as cv



prev_height = None

new_height = None


# define a video capture object 
vid = cv.VideoCapture(0) 

while(True): 

	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 
	cv.imwrite("output.jpg", frame)

	print(isObjectPresent("output.jpg", "bottle"))

	if isObjectPresent("output.jpg", "bottle"):
		quad, width, height = detectWithSearchName_v2("output.jpg", "bottle")
		#print(quad, width, height)

		if prev_height is None:
			prev_height = height
			new_height = height
		else:
			prev_height = new_height
			new_height = height
			closenessHelper(magnifier(prev_height, new_height),quad)


	#frame = cv.imread("result.jpg")

	# Display the resulting frame 
	#cv.imshow('frame', frame) 



	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv.waitKey(1) == ord('q'):
		break
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 
