import cv2
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your image file
image_path = 'image.jpg'

# Read the image using OpenCV
img = cv2.imread(image_path)
U=np.array([])
S=np.array([])
Vt=np.array([])
# Func to Show Image
def show_image(img):
    if img is not None:
        # Display the image using OpenCV
        new_size = (500, 500)
        r_img = cv2.resize(img, new_size)

        cv2.imshow('Image', r_img)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the image window

    else:
        print(f"Error: Unable to load image from {image_path}")

def calculate_svd(A):
  U, S, Vt = np.linalg.svd(A)
  return U,S,Vt
  

def kth_rank_app(k):
  U_k = U[:, :k]
  Sigma_k = np.diag(S[:k])
  Vt_k = Vt[:k, :]
  A_k = U_k @ Sigma_k @ Vt_k
  return A_k

def low_rank_approximation():
  # U, S, Vt = np.linalg.svd(A)

  plt.figure(figsize=(20, 8))
  k=1
  for i in range(0,12):
    k=k+i*5
    a_k=kth_rank_app(k)
    plt.subplot(2, 6, i+1)
    new_size = (500, 500)
    resized_img = cv2.resize(a_k, new_size)
    plt.imshow(resized_img)
    plt.title(f'Image for rank {k}')
    plt.axis('off')
  plt.tight_layout()
  plt.show()

def lowest_kth_rank_img(k):
  # U, S, Vt = np.linalg.svd(A)
  a_k=kth_rank_app(k)
  new_size = (500, 500)
  resized_img = cv2.resize(a_k, new_size)
  plt.imshow(resized_img)
  plt.title(f'Image for rank {k}')
  plt.axis('off')
  plt.show()
   

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
# calculate_svd(gray_img)
U,S,Vt=calculate_svd(gray_img)
# show_image(gray_img)
# low_rank_approximation()
lowest_kth_rank_img(60)

