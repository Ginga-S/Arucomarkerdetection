import cv2

img = cv2.imread('Sample Resources/lena.jpg', 0)
#second argument is a flag
#loads images in: 1=color, 1=grayscale, -1=as it is including alpha channel

print(img)

cv2.imshow('image', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()