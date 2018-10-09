import numpy as np
import cv2 as cv
import pickle
import time
import os
import math
import time
import matplotlib.pyplot as plt

def colordist(x, v):
    px = x[0]**2 + x[1]**2 + x[2]**2
    pv = v[0]**2 + v[1]**2 + v[2]**2
    px_pv = (x[0]*v[0] + x[1]*v[1] + x[2]*v[2])**2
    p = 0 if pv == 0 else px_pv/pv
    return math.sqrt(abs(px - p))

def brightness(I, Imin, Imax, alpha = 0.4, beta = 1.1):
    I_high = min(beta * Imax, Imin / alpha)
    I_low = alpha * Imax
    if I_low <= I and I <= I_high:
        return True
    else:
        return False

# https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
def codebookSave(codebooks):
    with open("codebook.pickle", "wb") as handle:
        pickle.dump(codebooks, handle, protocol=pickle.HIGHEST_PROTOCOL)

def codebookLoad():
    with open('codebook.pickle', 'rb') as handle:
        return pickle.load(handle)

def toFloat(pixel):
    return [float(pixel[0]), float(pixel[1]), float(pixel[2])]

def codebookCreate(height, width):
    """ Recebe altura e largura da imagem para criar codebook """
    # Create a empty codebood (I)
    codebooks = []
    for i in range(height):
        codebooks.append([])
        for j in range(width):
            codebooks[i].append([])
    return codebooks

def codebookUpdate(pixel, codeword_list, t, e1 = 12):
    pixel = toFloat(pixel)
    match_codeword = None
    match_index = None

    # Create brightness (II.I)
    I = np.linalg.norm(pixel)

    # Find matching codeword (II.II)
    if codeword_list:
        for index, codeword in enumerate(codeword_list):
            if (colordist(pixel, codeword[0]) <= e1 and 
                brightness(I, codeword[1][0], codeword[1][1])):
                match_codeword, match_index = codeword, index
                break

    # Create a new codeword (II.III)
    if not codeword_list or match_codeword == None:
        new_codeword = [pixel, [I, I, 1, t-1, t, t]]
        codeword_list.append(new_codeword)

    # Update matched codeword (II.IV)
    else:
        Vm = match_codeword[0] #Vm = [Rm, Gm, Bm] 
        AUXm = match_codeword[1] #AUXm = [Im_min, Im_max, Fm, LAMBm, Pm, Qm]
        Vm_u = [(AUXm[2] * Vm[0] * pixel[0])/AUXm[2] + 1, (AUXm[2] * Vm[1] * pixel[1])/AUXm[2] + 1, (AUXm[2] * Vm[2] * pixel[2])/AUXm[2] + 1]
        AUXm_u = [min(I, AUXm[0]), max(I, AUXm[1]), AUXm[2] + 1, max(AUXm[3], t - AUXm[5]), AUXm[4], t]
        codeword_list[match_index] = [Vm_u, AUXm_u]

    return codeword_list

def updateLambda(codeword_list, N):
    # Update Lambda on each codeword (III) 
    for codeword in codeword_list:
        new_lambda = max(codeword[1][3], (N - codeword[1][5] + codeword[1][4] - 1))
        codeword[1][3] = new_lambda

def codebookConstruct(codebooks, image, t):
    height, width, s = image.shape
    start = time.time()
    for i in range(height):
        for j in range(width):
            codebooks[i][j] = codebookUpdate(image[i][j], codebooks[i][j], t)
    
    for i in range(height):
        for j in range(width):
            updateLambda(codebooks[i][j], len(codebooks[i][j]))
    
    codebookSave(codebooks)

    end = time.time()
    print("Imagem {} done in {} seconds".format(t+1, (end-start)))
    
def codebookTraining(size):
    image_dir = "/home/martimfj/Downloads/ImagensVisao/S0_BG/Crowd_PETS09/S0/Background/View_001/Time_13-38/"
    image_file_list = os.listdir(image_dir)
    np.random.shuffle(image_file_list)
    image = cv.imread(image_dir + "00000001.jpg")
    height, width, s = image.shape
    codebooks = codebookCreate(height, width)

    for index, image_file in enumerate(image_file_list[:size]):
        image = cv.imread(image_dir + image_file)
        codebookConstruct(codebooks, image, index)

train_codebooks = False
if train_codebooks:
    start = time.time()
    codebookTraining(15)
    codebooks = codebookLoad()
    end = time.time()
    print("Total training time: {} seconds".format(end-start))
else:
    codebooks = codebookLoad()

def foregroundDetector(pixel, codeword_list, e2 = 8):
    pixel = toFloat(pixel)
    I = np.linalg.norm(pixel)

    for codeword in codeword_list:
        if (colordist(pixel, codeword[0]) <= e2 and 
            brightness(I, codeword[1][0], codeword[1][1])):
            return False
    return True

def backgroundSubtraction(image, codebooks):
    height, width, s = image.shape
    image_no_bg = image.copy()

    for i in range(height):
        for j in range(width):
            if foregroundDetector(image[i][j], codebooks[i][j]):
                image_no_bg[i][j] = [255, 255, 255]
            else:
                image_no_bg[i][j] = [0, 0, 0]
    return image_no_bg

image_dir = "/home/martimfj/Downloads/ImagensVisao/S1_L1/Crowd_PETS09/S1/L1/Time_13-57/View_001/frame_0000.jpg"
image_test = cv.cvtColor(cv.imread(image_dir), cv.COLOR_BGR2RGB)
plt.imshow(image_test)
plt.show()

result = backgroundSubtraction(image_test, codebooks)
plt.imshow(result)
plt.show()

