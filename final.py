import cv2
import numpy as np



pts = np.zeros((4, 2), dtype=np.float32)
capture = cv2.VideoCapture("driving.mp4")

while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, img1 = capture.read()
    img = img1.copy()

    cv2.circle(img1,(100,210),5,(0,0,255),-1)
    cv2.circle(img1, (5, 290), 5, (0, 0, 255), -1)
    cv2.circle(img1, (400, 210), 5, (0, 0, 255), -1)
    cv2.circle(img1, (480, 270), 5, (0, 0, 255), -1)


    #위 왼 오 아래 왼 오 순순

    pts15=np.float32([[130,210],[390, 210],[10, 290],[470, 270]])
    #변형된 파일 사이즈  윈온 윈오 순
    pts25=np.float32([[0,0],[400,0],[0,600],[400,600]])

    matrix=cv2.getPerspectiveTransform(pts15,pts25)
    result=cv2.warpPerspective(img1,matrix,(400,600))
    cv2.imshow('original',img1)
    cv2.imshow('scanned', result)


    ###############
    SIZE=500





    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect edges
    edges = cv2.Canny(gray, 150, 300)
    cv2.imshow("d",edges)

    # create mask
    mask = np.zeros(img.shape[:2], dtype="uint8")  # 0 - 255 = 8 bits

    # white pentagon
    pts = np.array([[0, SIZE * 3 / 5], [SIZE/3 , SIZE/3 ], [SIZE, SIZE * 2 / 5], [SIZE, SIZE], [0, SIZE]], np.int32)

    # black triangle
    pts2 = np.array([[SIZE / 2, 0], [SIZE / 4, SIZE], [SIZE * 2/ 4, SIZE]], np.int32)

    cv2.fillPoly(mask, [pts], 255)

    cv2.fillPoly(mask, [pts2], 0)
    cv2.imshow("x",mask)
    line_img = cv2.bitwise_and(edges, mask)
    cv2.imshow("a",line_img)

    # get lines
    # (x1, y1, x2, y2)
    lines = cv2.HoughLinesP(
        line_img,
        rho=1.0,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,
        maxLineGap=10
    )


    # get average line 임계값사용하고 싶어서 0.8프로의 것을 사용한다,
    def get_length_threshold(lines):
        lengths = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            # append는 그 리스트의 맨 마지막 부분의 인자요소를 더해준다.
        # set threshold to top 80% longest lines
        return np.quantile(lengths, 0.97)
        # 백분인수 0.8을 사용했는데 만약 0.5를 사용하면 가장 가운데 값이다. 0.8이니까 순서대로 나열햇울때 80%의 위치의 값 호출


    left_counter = right_counter = 0
    left_x, left_y, right_x, right_y = [], [], [], []
    length_threshold = get_length_threshold(lines)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # for every line

            if x1 == x2: continue  # to avoid division by 0
            # if continue를 사용하여 이 것일 때 건너 뛴다는 것이다.
            # in code, y is positive down
            slope = (y1 - y2) / (x2 - x1)
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # ensure only long lines are considered(긴 줄만 고려되는지 확인) 여기서 길이를 우리가 지정할수있지만 임계값을 사용하겟다.
            if length < length_threshold: continue

            # these coords belong to right line(오른쪽 좌표에 속한다.)
            if slope < 0:  # 오른쪽 선은 \모양이라 음수니까
                right_counter += 1
                right_x += [x1, x2]
                right_y += [y1, y2]
            # left line
            else:
                left_counter += 1
                left_x += [x1, x2]
                left_y += [y1, y2]

            # calculate linear fit
    BOTTOM_Y = img.shape[0] * 3 // 4
    # 소수라인 더 적은 방식
    # img.shape[0] 높이, 1 너비 2는 층
    # top_y는 0 이어야되는데 그 선의 범위를 줄이고 싶고 소수점줄이고 싶어서 뒤에 *3//5를 붙인것이다.
    TOP_Y = img.shape[0] * 3 // 6
    LANE_COLOR = (0, 255, 0)


    def draw_average_line(x_list, y_list, counter):
        if counter > 0:
            polyfit = np.polyfit(y_list, x_list, deg=1)
            # 맨마지막 1이니까 1차함수 즉 직선의 방정식을 구해줌 1차식의 계수를 구해준다.
            poly = np.poly1d(polyfit)
            # poly를 통해 완벽한 방정식을 만든다.
            print(poly)
            x_start = int(poly(BOTTOM_Y))
            x_end = int(poly(TOP_Y))
            cv2.line(img, (x_start, BOTTOM_Y), (x_end, TOP_Y), LANE_COLOR, 5)


    # draw average line 위 함수 사용
    draw_average_line(left_x, left_y, left_counter)
    draw_average_line(right_x, right_y, right_counter)


    # 위 왼 오 아래 왼 오 순순

    pts115 = np.float32([[130, 210], [390, 210], [10, 290], [470, 270]])
    pts215 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

    matrix = cv2.getPerspectiveTransform(pts115, pts215)
    result2 = cv2.warpPerspective(img, matrix, (400, 600))

    cv2.imshow("Image", result2)
    cv2.imshow("22", img)


capture.release()
cv2.destroyAllWindows()