import cv2
import os
import glob

def crop_and_save(image, output_path, num):
    img = cv2.imread(image)
    height, width, channels = img.shape
    h = int((height * 25) / 100)
    x, y = 10, 1
    crop_img = img[y:h, x:width]
    cv2.imwrite(os.path.join(output_path, f"{num}.png"), crop_img)

def main():
    files = glob.glob("cluster/*")
    num = 0

    for file in files:
        num = 0
        c_file = os.path.join(file, 'cropped')
        os.makedirs(c_file, exist_ok=True)

        images = glob.glob(os.path.join(file, '*.png'))
        
        for image in images:
            num += 1
            crop_and_save(image, c_file, num)
            # Uncomment the lines below if you want to show each cropped image
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)

if __name__ == "__main__":
    main()
