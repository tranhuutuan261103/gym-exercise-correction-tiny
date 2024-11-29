from plank_model import PlankModel
import cv2

model = PlankModel()

image_path = "test.jpg"

image = cv2.imread(image_path)

result_image = model.predict(frame=image, prediction_probability_threshold=0.5)

cv2.imshow("Result", result_image)
cv2.waitKey(0)