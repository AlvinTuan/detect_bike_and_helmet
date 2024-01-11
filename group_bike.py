import cv2
import os
import numpy as np
from skimage.io import imread_collection

def save_image(path, image, sub_ele):
    cv2.imwrite(os.path.join(path, f"{sub_ele}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main():
    images = imread_collection('./images/*.png')
    print(images)
    
    skip = 0
    ok = []
    not_ok = []
    sub_ele = 0
    bike_num = 0
    first_img = images[0]
    
    newpath = f'cluster/bike_{bike_num}'
    os.makedirs(newpath, exist_ok=True)
    
    save_image(newpath, first_img, sub_ele)
    sub_ele += 1
    num = 0
    ok.append(num)
    correct = 0
    
    while True:
        num += 1
        if num == len(images):
            print("Done with 'Grouping_Images'")
            break
        
        second_img = images[num]
        
        # 1) Check if 2 images are equals
        if np.array_equal(first_img, second_img):
            print("The images are completely Equal")
        else:
            print("The images are NOT equal")
        
        # 2) Check for similarities between the 2 images
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(first_img, None)
        kp_2, desc_2 = sift.detectAndCompute(second_img, None)
        
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_1, desc_2, k=2)
        good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]
        
        print("total good_points: ", len(good_points))
        
        if len(good_points) >= 10:
            print("correct number: ", num)
            save_image(newpath, second_img, sub_ele)
            sub_ele += 1
            ok.append(num)
            correct = num
            skip += 1
            wrong_preds = 0
        else:
            wrong_preds += 1
            skip += 1
            
            if num not in ok:
                not_ok.append(num)
            
            if wrong_preds > 1:
                print("wrong preds: ", wrong_preds)
                print("total skips: ", skip)
                print("current number: ", num)
                print("user cluster is completed")
                print("ok list: ", ok)
                print("not_ok: ", not_ok)
                print("*******************")
                
                sub_ele = 0
                bike_num += 1
                first_img = images[not_ok[0]]
                num = not_ok[0]
                
                print("update number(first): {}".format(num))
                not_ok = []
                newpath = f'cluster/bike_{bike_num}'
                os.makedirs(newpath, exist_ok=True)
                
                save_image(newpath, first_img, sub_ele)
                ok.append(num)
                sub_ele += 1
                skip = 0
                wrong_preds = 0
            else:
                print("user cluster is 'not completed'")

if __name__ == "__main__":
    main()
