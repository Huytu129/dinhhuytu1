import cv2
import numpy as np

# Đường dẫn ảnh gốc
image_path = 'anh-viet-nam.jpg'

# Đọc ảnh gốc dưới dạng ảnh xám (grayscale)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Kiểm tra xem ảnh có được đọc thành công không
if image is None:
    print("Không thể đọc ảnh, kiểm tra lại đường dẫn.")
    exit()

# 1. Ảnh âm tính (Negative Image)
negative_image = 255 - image
cv2.imwrite('negative_image.jpg', negative_image)

# 2. Tăng độ tương phản (Contrast Enhancement)
alpha = 2.0  # Hệ số tăng cường (có thể điều chỉnh)
beta = 50    # Giá trị offset (có thể điều chỉnh)
contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imwrite('contrast_image.jpg', contrast_image)

# 3. Biến đổi log (Log Transformation)
# Chuyển đổi ảnh sang float và thêm giá trị nhỏ để tránh log(0)
image_float = np.float32(image)  # Chuyển sang float
image_float += 1  # Thêm 1 để tránh log(0)

# Thực hiện phép biến đổi log
log_image = np.log(image_float)

# Chuẩn hóa giá trị về khoảng 0-255 và chuyển về uint8
log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)
log_image = np.uint8(log_image)
cv2.imwrite('log_image.jpg', log_image)

# 4. Cân bằng Histogram (Histogram Equalization)
hist_eq_image = cv2.equalizeHist(image)
cv2.imwrite('hist_eq_image.jpg', hist_eq_image)

print("Đã xử lý xong các thao tác và lưu ảnh kết quả.")