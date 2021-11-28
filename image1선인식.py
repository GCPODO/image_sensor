import cv2
#opencv2 사용 시작
import numpy as np
#행렬 함수 사용하고 np로 사용하겠다.

img = cv2.imread("road.jpg")
#img에  road의 이미지 파일 들어보겠다.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray 변수에 cvtColor(이미지파일,무슨색으로할건지 마지막 그래이니까 회색)

# detect edges
edges = cv2.Canny(gray, 150, 300)
#이미지 파일에서 그 사진의 윤곽만 잡아서 그림 파일로 만들어주는 cv2.canny 150의 값이 커지면 윤곽이 더 없어짐

# get lines
# (x1, y1, x2, y2)
lines = cv2.HoughLinesP(
    #이미지 선모양또느 값에따라 원같은 다양한 모양인식가능 도형을 검출하는 Hough Transform 이다.

    edges,
    rho=1.0,
    #거리측정 해상도 0~1
    theta=np.pi/180,
    #각도 정밀 pi/0~180
    threshold=20,
    #직선으로 판단할 최소한의 동일 개수 작은값: 정확도감소, 증가:정확도증가 프로그래밍 부화
    minLineLength=30,
    # 선으로 인정할 최소 길이
    maxLineGap=10
    # 선으로 판단할 최대간격
)

# draw lines 선그리기 녹색선
line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#행렬0 을가진 img 행 ,img열,3층, dtype=np.uint8  데이터 파일 0~255의 형태로 변환한다.
#BGR이니까 3층 쓰는거임

line_color = [0, 255, 0]
#BGR 순이니까 255 green 으로 선 검출하겠다고 선언
line_thickness = 2
#선두께
dot_color = [0, 255, 0]
dot_size = 3
#점 크기
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
        #선그리기 파일
        cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
        #원그리기 마지막 -1이면 원 안 채움
        cv2.circle(line_img, (x2, y2), dot_size, dot_color, -1)

overlay = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#두개의 영상을 합할때 어느 사진을 더 선명하게 할지 정한다.
#여기서는 img 0.8주어서 line img는 1을 주어서 img가 더 흐리게 나옴 0.0은 불필요
cv2.imshow("Overlay", overlay)
#윈도우 창에서 사진 보여준다.
cv2.waitKey()
#이 키값 입력 될수도 잇따.

cv2.destroyAllWindows()
#우리가 생성한 모든 윈도우를 제거한다.