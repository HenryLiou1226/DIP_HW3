import cv2 
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
name = 'test'
sign = cv2.imread('sign.png', cv2.IMREAD_UNCHANGED)
image = cv2.imread(f'{name}.png')
def canny_edge_detector(image): 
     # 先轉灰階
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    # 高斯模糊化
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 計算 Sobel X 和 Sobel Y
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # 計算梯度幅值
    image = np.sqrt(sobelx**2 + sobely**2)
    # 計算梯度方向 tan < 0 = tan + 180
    gradient = np.arctan2(sobely, sobelx) * 180 / np.pi
    gradient[gradient < 0] += 180
    temp_image = np.zeros_like(image)
    # Non-maximum suppression
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (0 <= gradient[i, j] < 22.5) or (157.5 < gradient[i, j] <= 180):
                if (j + 1 < image.shape[1] and image[i, j] > image[i, j + 1]) and (j - 1 >= 0 and image[i, j] > image[i, j - 1]):
                    temp_image[i, j] = image[i, j]
            elif (22.5 <= gradient[i, j] <= 67.5):
                if (i - 1 >= 0 and j + 1 < image.shape[1] and image[i, j] > image[i - 1, j + 1]) and (i + 1 < image.shape[0] and j - 1 >= 0 and image[i, j] > image[i + 1, j - 1]):
                    temp_image[i, j] = image[i, j]
            elif (67.5 < gradient[i, j] < 112.5):
                if (i - 1 >= 0 and image[i, j] > image[i - 1, j]) and (i + 1 < image.shape[0] and image[i, j] > image[i + 1, j]):
                    temp_image[i, j] = image[i, j]
            elif (112.5 <= gradient[i, j] <= 157.5):
                if (i - 1 >= 0 and j - 1 >= 0 and image[i, j] > image[i - 1, j - 1]) and (i + 1 < image.shape[0] and j + 1 < image.shape[1] and image[i, j] > image[i + 1, j + 1]):
                    temp_image[i, j] = image[i, j]
    image = temp_image
    # 雙門檻和連通成份分析 thresholding low 100 high 200
    low =  20
    high = 60
    # 連通成份分析
    queue = deque()
    # 紀錄強像素
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] >= high:
                image[i, j] = 255
                queue.append((i, j))
            elif image[i, j] < low:
                image[i, j] = 0
            else:
                image[i, j] = low
    while queue:
        x, y = queue.popleft()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x + i < 0 or x + i >= image.shape[0] or y + j < 0 or y + j >= image.shape[1]:
                    continue
                # 連接到弱像素時，將弱像素設為強像素，並加入queue
                if image[x + i, y + j] == low:
                    image[x + i, y + j] = 255
                    queue.append((x + i, y + j))
    # 將非強像素設為0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] != 255:
                image[i, j] = 0
    return image

def hough_lines(img,threshold):
    # 圖形x,y平面大小
    height, width = img.shape[0], img.shape[1]
    # p的最長值為最長邊長根號2 cos(45) + sin(45) = 1/根號2
    p = int(max(height, width) * np.sqrt(2))

    # rho範圍為-p到p，θ範圍為-90到90
    max_rho = p
    max_theta = 90

    # 紀錄ρθ平面的累加器
    accumulator = np.zeros((2 * max_rho, 2 * max_theta))

    # 掃x,y平面，不為零的線畫-90 < ϴ < 90的線，記錄到ρθ平面的點
    for x in range(height):
        for y in range(width):
            if img[x][y] > 0:
                # 若x,y平面為線，轉換到ρθ平面的點 -90 < ϴ < 90
                for t in range(-max_theta, max_theta):
                    r = int(y * np.cos(np.deg2rad(t)) + x * np.sin(np.deg2rad(t)))
                    if -max_rho <= r < max_rho:
                        if t + max_theta - 1 >= 0:
                            accumulator[r + max_rho][t + max_theta - 1] += 1
                        if r + max_rho - 1 >= 0:
                            accumulator[r + max_rho - 1][t + max_theta] += 1
                        if t + max_theta + 1 < 2 * max_theta:
                            accumulator[r + max_rho][t + max_theta + 1] += 1
                        if r + max_rho + 1 < 2 * max_rho:
                            accumulator[r + max_rho + 1][t + max_theta] += 1
                        accumulator[r + max_rho][t + max_theta] += 1
    # θ和ρ的分布圖
    plt.figure(figsize=(10, 10))
    plt.imshow(accumulator, cmap='hot', extent=[-max_theta, max_theta, -max_rho, max_rho],aspect='auto')
    plt.title('accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.savefig(f'accumulator_{name}.png')
    # 紀錄直線
    lines = []
    for r in range(2 * max_rho):
        for t in range(2 * max_theta):
            # 根據ϴρ點轉換回xy平面直線
            if accumulator[r][t] > threshold:
                rho_val = r - max_rho
                theta_val = t - max_theta
                x0 = rho_val * np.cos(np.deg2rad(theta_val))
                y0 = rho_val * np.sin(np.deg2rad(theta_val))
                # xcos(ϴ) + ysin(ϴ) = ρ -> (x + 10000(sin(ϴ))cos(ϴ)) + (y + 10000(-cos(ϴ))sin(ϴ)) = ρ 劃出直線
                x1 = int(x0 + 10000 * (-np.sin(np.deg2rad(theta_val))))
                y1 = int(y0 + 10000 * np.cos(np.deg2rad(theta_val)))
                x2 = int(x0 - 10000 * (-np.sin(np.deg2rad(theta_val))))
                y2 = int(y0 - 10000 * np.cos(np.deg2rad(theta_val)))
                lines.append(((x1, y1), (x2, y2)))
    return lines
canny_image = canny_edge_detector(image)
lines = hough_lines(canny_image,400)
hough_image = canny_image.copy()
for line in lines:
    cv2.line(hough_image, line[0], line[1], (255, 255, 255), 2)
for i in range(sign.shape[0]):
    for j in range(sign.shape[1]):
        if sign[i,j,3] != 0:
            canny_image[canny_image.shape[0] - sign.shape[0] + i - 1,canny_image.shape[1] - sign.shape[1] + j - 1] = sign[i,j,0]
for i in range(sign.shape[0]):
    for j in range(sign.shape[1]):
        if sign[i,j,3] != 0:
            hough_image[hough_image.shape[0] - sign.shape[0] + i - 1,hough_image.shape[1] - sign.shape[1] + j - 1] = sign[i,j,0]
cv2.imwrite(f'canny_image_{name}.png', canny_image)
cv2.imwrite(f'Hough Lines_{name}.png', hough_image)