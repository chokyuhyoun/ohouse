import pyautogui as pag
import numpy as np
import mss, cv2, win32gui, win32ui, win32con, pdb, time, mouse
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp

def check_num(color):
    global color_list
    color_list = np.array([[255, 246, 222], # 0 
                            [255, 254, 255], # 2
                            [255, 236, 170], # 4
                            [255, 229, 143], # 8
                            [255, 216,  87], # 16
                            [241, 183,   0], # 32
                            [204, 140,   1], # 64
                            [ 60, 240, 141], # 128
                            [ 32, 232, 109], # 256z
                            [ 26, 222,  77], # 512
                            [121, 225, 255], # 1024
                            [ 13, 201, 255], # 2048 
                            [  1, 151, 255]]) # 4096
    color = color.flatten()
    loc = np.argmin(np.sum((color_list - color)**2, axis=1))
    return loc.item()

def get_tile_array(tile_xpos, tile_ypos):
    tile_array = np.zeros((4, 4))
    for ii in range(4):
        for jj in range(4):
            with mss.mss() as sct:
                color = np.array(sct.grab(
                    {'left':tile_xpos[ii], 'top':tile_ypos[jj], 
                     'width':1, 'height':1}))[:, :, :3]
                pass
            # print(color)
            tile_array[jj, ii] = check_num(color)
    return tile_array

def after_moving(tile_array, direction='up'):
    if direction == 'up':
        tile_array0 = copy.deepcopy(tile_array)
    if direction == 'down':
        tile_array0 = copy.deepcopy(tile_array[::-1])
    if direction == 'right':
        tile_array0 = copy.deepcopy(tile_array.T[::-1])
    if direction == 'left':
        tile_array0 = copy.deepcopy(tile_array.T)

    templet = np.zeros((4, 4))
    for i in range(4): # summing
        part = tile_array0[:, i]
        part0 = part[part > 0]
        for j in range(len(part0)-1):
            if part0[j] == part0[j+1]:
                part0[j] += 1
                part0[j+1] = 0
        part1 = part0[part0 > 0]
        templet[:len(part1), i] = part1

    if direction == 'down':
        templet = templet[::-1]
    if direction == 'right':
        templet = templet[::-1].T
    if direction == 'left':
        templet = templet.T
    return templet

def evaluation(after_tile_array):
    table0 = np.array([[1, 2, 3, 4], 
                       [4, 5, 6, 7], 
                       [7, 8, 9, 10], 
                       [10, 11, 12, 13]])
    # table0 = np.arange(4*4)[::-1].reshape(4, 4)    
    table = 2**(table0)
    # table[1] = table[1][::-1]
    # table[3] = table[3][::-1]

    # table[0, 0] *= 10
    # dum = after_tile_array
    dum = np.where(after_tile_array==0, 2, after_tile_array)
    return np.sum(table*(2**dum))

def find_direct(tile_array, depth=3):
    import numpy as np
    if np.sum(tile_array >= 9) > 3: return 'up', -1
    if (depth % 2) == 1:
        dir_order = ['left', 'up', 'right', 'down']
        score = np.zeros(len(dir_order))
        for ii, try_dir in enumerate(dir_order):
            after = after_moving(tile_array, direction=try_dir)
            if (after == tile_array).all(): 
                score[ii] = -depth
            elif depth == 1:
                score[ii] = evaluation(after)
            else:
                _, score[ii] = find_direct(after, depth=depth-1)
        return dir_order[np.argmax(score)], np.max(score)    
    else : #depth // 2 == 0:
        empty = np.where(tile_array == 0)
        dir_score = []
        for ii in range(len(empty[0])):
            tile_array[empty[0][ii], empty[1][ii]] = 1
            direct_sub, score_sub = find_direct(tile_array, depth=depth-1)
            dir_score.append(score_sub)
            # for jj in range(2):
            #     tile_array[empty[0][ii], empty[1][ii]] = jj + 1
            #     direct_sub, score_sub = find_direct(tile_array, depth=depth-1)
            #     fact = 0.9 if jj == 0 else 0.1
            #     dir_score.append(score_sub*fact)
        return 'none', np.mean(dir_score)

def find_direction(tile_array, depth=3):
    dir_order = ['left', 'up', 'right', 'down']
    after0 = []
    for direct in dir_order:
        after0.append(after_moving(tile_array, direct))
    # print(after0)
    # pdb.set_trace()
    args = list(zip(after0, [depth-1]*4, strict=True))
    with mp.Pool(processes=4) as pool:
        results = pool.starmap(find_direct, args)
    # max_ind = np.argmax(results[:, 1])
    # return dir_order[max_ind], results[max_ind, 1]
    return results
    # return dir_order[np.argmax(results)], np.max(results)    

def find_direction2(tile_array, depth=3):
    dir_order = ['left', 'up', 'right', 'down']
    # pool = mp.Pool(processes=4)
    # results = []
    # after = []
    # for direct in dir_order:
    #     after.append(after_moving(tile_array, direct))
    # for ii in range(4):
    #     results.append(pool.apply_async(find_direct, (after[ii], depth-1)))
    # results = [res.get() for res in results]
    with Pool(processes=4) as pool:  # Adjust the number of processes based on your CPU
        results = pool.map(lambda move: find_direct(tile_array, depth-1), dir_order)

    return results


def drag(direction):
    delta = 30
    x, y = 220, 779
    if direction=='up': mouse.drag(x, y, x, y-delta)
    if direction=='down': mouse.drag(x, y, x, y+delta)    
    if direction=='right': mouse.drag(x, y, x+delta, y)
    if direction=='left': mouse.drag(x, y, x-delta, y)
