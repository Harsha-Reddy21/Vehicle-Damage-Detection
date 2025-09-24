import cv2
import numpy as np

class ImageQualityAgent:
    def __init__(self, quality_threshold=0.6):
        self.quality_threshold = quality_threshold

    # --- 1. Image quality assessment ---
    def estimate_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def estimate_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness / 255.0

    def estimate_resolution(self, image):
        height, width = image.shape[:2]
        return width * height

    def detect_issues(self, image):
        issues = []
        if self.estimate_blur(image) < 100:  # threshold for blur
            issues.append("blur")
        if self.estimate_brightness(image) < 0.4:
            issues.append("low_light")
        if self.estimate_resolution(image) < 200*200:  # example: reject < 200x200
            issues.append("low_resolution")
        return issues

    # --- 2. Image enhancement ---
    def enhance_image(self, image):
        # Convert to LAB for histogram equalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])  # L channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Optional sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    # --- 4. Basic image manipulation detection ---
    def detect_manipulation(self, image):
        # Placeholder: simple method using noise patterns
        # Advanced: use pretrained forgery detection models
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_std = np.std(gray)
        # Extremely uniform or overly noisy images could be suspicious
        if noise_std < 5 or noise_std > 80:
            return True
        return False

    # --- Main process ---
    def process(self, image):
        issues = self.detect_issues(image)
        enhanced_img = self.enhance_image(image)
        quality_score = 1.0 - len(issues)*0.2  # simple scoring
        processable = quality_score >= self.quality_threshold
        manipulation_detected = self.detect_manipulation(image)

        return {
            "quality_score": round(quality_score, 2),
            "issues": issues,
            "enhanced_image": enhanced_img,
            "processable": processable,
            "manipulation_detected": manipulation_detected
        }

# ------------------------------
# Demo
# ------------------------------
if __name__ == "__main__":
    image_path = "C:/Misogi/vehicle_dataset/C9LJUWLH/car-images/front_left-15.jpeg" # replace with your image
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image. Check the path!")
        exit()

    agent = ImageQualityAgent()
    result = agent.process(image)

    print("Quality Score:", result["quality_score"])
    print("Issues Detected:", result["issues"])
    print("Processable:", result["processable"])
    print("Manipulation Detected:", result["manipulation_detected"])

    # Show original vs enhanced images side by side
    combined = np.hstack((image, result["enhanced_image"]))
    cv2.imshow("Original (Left) vs Enhanced (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
