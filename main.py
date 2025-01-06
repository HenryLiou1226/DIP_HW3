import cv2 
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
name = 'pic3'
# 因為要讀透明層 所以unchanged 
sign = cv2.imread('sign.png', cv2.IMREAD_UNCHANGED)
image = cv2.imread(f'{name}.jpg')
def canny_edge_detector(image): 
     # 先轉灰階
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    # 5x5 高斯模糊化
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 計算 Sobel X 和 Sobel Y 64F 因為可能會有負值
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # 計算梯度幅值
    image = np.sqrt(sobelx**2 + sobely**2)
    # 計算梯度方向 arctan2 return 範圍[-π, π] tan < 0 = tan + 180  
    gradient = np.arctan2(sobely, sobelx) * 180 / np.pi
    gradient[gradient < 0] += 180
    # 生成一個跟image大小相同的0陣列
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
    # 雙門檻和連通成份分析 thresholding low,high 
    low =  60
    high = 120
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
        # 八連通
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
    # p的最長值為最長邊長根號2 cos(45) + sin(45) = 2*(1/根號2)
    p = int(max(height, width) * np.sqrt(2))

    # rho範圍為-p到p，θ範圍為0到180
    max_rho = p
    max_theta = 180

    # 紀錄ρθ平面初始化為0的累加器 size = (2 * max_rho + 1)* (max_theta + 1) 
    accumulator = np.zeros((2 * max_rho + 1, max_theta + 1))

    # 掃x,y平面，不為零的點畫0 < ϴ < 180的線，記錄到ρθ平面的點
    for x in range(height):
        for y in range(width):
            if img[x][y] > 0:
                # 若x,y平面為線，轉換到ρθ平面的點 0 < ϴ < 180
                for t in range(0, max_theta + 1):
                    r = int(y * np.sin(np.deg2rad(t)) + x * np.cos(np.deg2rad(t)))
                    # 別只投剛好的，附近也投
                    if t + max_theta - 1 >= 0:
                        accumulator[r + max_rho][t - 1] += 1
                    if r + max_rho - 1 >= 0:
                        accumulator[r + max_rho - 1][t] += 1
                    if t + max_theta + 1 <= max_theta:
                        accumulator[r + max_rho][t + 1] += 1
                    if r + max_rho + 1 <= 2 * max_rho:
                        accumulator[r + max_rho + 1][t] += 1
                    accumulator[r + max_rho][t] += 1
    # θ和ρ的分布圖
    # 10x10 auto -> 方形 hot -> 大小顏色不同區分
    plt.figure(figsize=(10, 10))
    plt.imshow(accumulator, cmap='hot', extent=[0, max_theta, max_rho, -max_rho],aspect='auto')
    plt.title('accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.savefig(f'accumulator_{name}.png')
    # 紀錄直線
    lines = []
    for r in range(2 * max_rho + 1):
        for t in range(max_theta + 1):
            # 根據ϴρ點轉換回xy平面直線
            if accumulator[r][t] > threshold:
                rho_val = r - max_rho
                theta_val = t
                x0 = rho_val * np.cos(np.deg2rad(theta_val))
                y0 = rho_val * np.sin(np.deg2rad(theta_val))
                # xcos(ϴ) + ysin(ϴ) = ρ ->  p cos(ϴ) ^ 2 + p sin(ϴ) ^ 2 = ρ 
                # (x + 10000(sin(ϴ))cos(ϴ)) + (y - 10000(-cos(ϴ))sin(ϴ)) = ρ 劃出直線
                x1 = int(x0 + 10000 * (-np.sin(np.deg2rad(theta_val))))
                y1 = int(y0 + 10000 * np.cos(np.deg2rad(theta_val)))
                x2 = int(x0 - 10000 * (-np.sin(np.deg2rad(theta_val))))
                y2 = int(y0 - 10000 * np.cos(np.deg2rad(theta_val)))
                lines.append(((y1, x1), (y2, x2)))
    return lines
canny_image = canny_edge_detector(image)
# pic 1 threshold = 350 pic 2 threshold = 400 pic 3 threshold = 300
lines = hough_lines(canny_image,300)
# copy一份canny_image來進行操作 直接a = b在python陣列中會變動到原本的陣列
hough_image = canny_image.copy()
for line in lines:
    # 畫直線 (255, 255, 255)白色 2線寬
    cv2.line(hough_image, line[0], line[1], (255, 255, 255), 2)
# 將sign加到canny_image和hough_image，透明度不為0的部分加到圖片上
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