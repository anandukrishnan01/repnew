import cv2
import os
from cvzone.PoseModule import PoseDetector

# Load person image
personImagePath = r"C:\Users\Anandu\Desktop\virtual_try\ng1531.jpg"
personImage = cv2.imread(personImagePath)
if personImage is None:
    print("Error: Failed to load person image")
    exit()

personHeight, personWidth, _ = personImage.shape

# Initialize PoseDetector
detector = PoseDetector()

# Path to shirt images
shirtFolderPath = r"C:\Users\Anandu\Desktop\virtual_try\shirt"
if not os.path.isdir(shirtFolderPath):
    print("Error: Shirt folder path is invalid")
    exit()

listShirts = os.listdir(shirtFolderPath)

# Calculate the fixed aspect ratio
fixedRatio = 262 / 190  # widthOfShirt / widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440

# Initialize image number for shirt selection
imageNumber = 0

# Initialize counters for button interaction (if needed)
counterRight = 0
counterLeft = 0
selectionSpeed = 4

while True:
    img = personImage.copy()  # Use a copy of the original image

    # Find poses in the image
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    if lmList:
        print("Landmark List:", lmList)  # Print landmark list to check

        # Calculate necessary parameters for shirt placement
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]
        
        # Load and resize the shirt image
        shirtImagePath = os.path.join(shirtFolderPath, listShirts[imageNumber])
        imgShirt = cv2.imread(shirtImagePath, cv2.IMREAD_UNCHANGED)
        if imgShirt is None:
            print("Error: Failed to load shirt image:", shirtImagePath)
            exit()

        # Calculate the width of the shirt based on pose landmarks
        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))

        # Calculate offset based on current scale
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        # Overlay the shirt on the person's body
        try:
            # Check if the shirt image is loaded successfully
            if imgShirt is not None:
                # Resize the shirt image if needed
                new_width = max(1, widthOfShirt)
                new_height = max(1, int(widthOfShirt * shirtRatioHeightWidth))
                imgShirt = cv2.resize(imgShirt, (new_width, new_height))

                # Overlay the resized shirt on the person's body
                for c in range(3):
                    img[lm12[1] - offset[1]:lm12[1] - offset[1] + imgShirt.shape[0],
                    lm12[0] - offset[0]:lm12[0] - offset[0] + imgShirt.shape[1], c] = \
                    img[lm12[1] - offset[1]:lm12[1] - offset[1] + imgShirt.shape[0],
                    lm12[0] - offset[0]:lm12[0] - offset[0] + imgShirt.shape[1], c] * \
                    (1 - imgShirt[:, :, 3] / 255.0)
        except Exception as e:
            print("Error:", e)

    cv2.imshow("Virtual Dressing Room", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
