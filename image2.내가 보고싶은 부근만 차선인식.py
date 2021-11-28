import cv2
import numpy as np


def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


SIZE = 500
img = cv2.imread("road4.png")
img = cv2.resize(img, (SIZE, SIZE))
#dst = cv2.resize(src, dstSize, fx, fy, interpolation)는 입력 이미지(src), 절대 크기(dstSize), 상대 크기(fx, fy), 보간법(interpolation)으로 출력 이미지(dst)을 생성합니다.
#cv2.resize 이미지를 500*500으로 변환시킨다. 자르는게 아니라 크기를 작게하는것이다.

show_image("Original", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect edges
edges = cv2.Canny(gray, 280, 310)
show_image("a",edges)

# create mask
mask = np.zeros(img.shape[:2], dtype="uint8")  # 0 - 255 = 8 bits
#shape[:2] 높이 너비 넣어줌, :3이면 층까지 넣어줌

# white pentagon
pts = np.array([[0, SIZE * 3 / 4], [SIZE / 2, SIZE / 2], [SIZE, SIZE * 3 / 4], [SIZE, SIZE], [0, SIZE]], np.int32)
#배열부분은 np.int32의 형식을 따른다. 이건 그냥 임의로 점들 정해준거임 깊은 생각 ㄴㄴ

# black triangle
pts2 = np.array([[SIZE / 2, 0], [SIZE / 4, SIZE], [SIZE * 3 / 4, SIZE]], np.int32)

cv2.fillPoly(mask, [pts], 255)
#여러점을 이은 블록 다각형을 그린다. pts는 좌표들 mask는 이미지 뒤에는 색깔
cv2.fillPoly(mask, [pts2], 0)

show_image("mask", mask)

# get lines
# (x1, y1, x2, y2)
lines = cv2.HoughLinesP(
    edges,
    rho=1.0,
    theta=np.pi / 180,
    threshold=20,
    minLineLength=30,
    maxLineGap=10
)

# draw lines
line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
line_color = [0, 255, 0]
line_thickness = 2
dot_color = [0, 255, 0]
dot_size = 3

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
        cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
        cv2.circle(line_img, (x2, y2), dot_size, dot_color, -1)

line_img = cv2.bitwise_and(line_img, line_img, mask=mask)
#라인이미지에서 mask의 이미지중 화이트 즉 1인 부분만 보겠다고 한것이다. 여기서는 and를 썼다. 즉 흰색과 흰색이 만나야 1이된다.
overlay = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#색 민감도 정리

show_image("Overlay", overlay)