#——————————————————————————————————————————————————————————————————————————————#
# This program takes any picture and fucks it up.                              #
#——————————————————————————————————————————————————————————————————————————————#


#—————————#
# Imports #
#—————————#
import os
from glob import glob
import numpy as np
from numba import jit ##@jit(nopython=True)
from PIL import Image as img
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog as fd

#———————————————————————————————————————————————————————————————————————————————


#———————————#
# FUNCTIONS #
#———————————#

#Prompts the user for an image file and some parameters, then generates a number
#of antichromatic images, animations, averages, and brings up the blending window
def main():
    intro()

    #Get inputs
    filepath,img_name,op,lums = get_file()
    n_th,n_sh,t_th,t_sh = get_slice_n(lums)

    #Generate shatters
    print("\n\n                    !!!Let's fuck it upp!!!\n\n  ",end='')
    batch,filepath = shatter(filepath,img_name,lums,n_th,n_sh,t_th,t_sh)
    h,w,chan = op.shape
    print("\n                  ***Antichromatizations generated!***\n\n")
    print(" *Attempting to gather meaning from this senseless act:\n ",end='')

    #Generate combos:
    combos = np.empty((h,w,16),dtype=np.uint8)
    combos[...,:8] = gen_combos(batch[...,:n_th],filepath,img_name + \
                            "-THRESH.png",op)
    combos[...,8:] = gen_combos(batch[...,n_th:],filepath,img_name + \
                            "-SHATT.png",op)
    
    print("\n\n\n                              ***Synthesizing***")
    print("                                      ...\n\n")
    
    #Put up graph
    nameo = ["THRESH: Mean","THRESH: Median","THRESH: Mode","THRESH: Outliers", \
             "THRESH: Mean_inv","THRESH: Median_inv","THRESH: Mode_inv",
             "THRESH: Outliers_inv", \
             "SHATT: Mean","SHATT: Median","SHATT: Mode","SHATT: Outliers", \
             "SHATT: Mean_inv","SHATT: Median_inv","SHATT: Mode_inv",
             "SHATT: Outliers_inv"]
    disp_mixer(combos,nameo,outpath=filepath,filename=img_name)

    outro()
    
#———————————————————————————————————————————————————————————————————————————————

#Calculates and outputs mean, median, mode, and outlier images, and their
#inverses, based on inputted batch
def gen_combos(batch,fp,imn,op):
    h,w,n = batch.shape
    specials = np.zeros((h,w,8),dtype=np.uint8)
    print("...",end='')
    
    #Mean & Inverse
    specials[...,0] = gen_mean(batch,filepath=fp,filename=imn)
    specials[...,4] = 255 - specials[...,0]
    print("?...",end='')
    #Outliers & Inverse
    specials[...,3] = gen_outliers(batch,op,filepath=fp,filename=imn)
    specials[...,7] = 255 - specials[...,3]
    print("?...",end='')
    #Median & Inverse
    specials[...,1] = gen_median(batch,filepath=fp,filename=imn)
    specials[...,5] = 255 - specials[...,1]
    print("?...",end='')
    #Mode & Inverse (make 'em sweat)
    specials[...,2] = gen_mode(batch,filepath=fp,filename=imn)
    specials[...,6] = 255 - specials[...,2]
    print("!",end='')

    return specials

#———————————————————————————————————————————————————————————————————————————————

#Prints header
def intro():
    print("\n********************************************")
    print("* Welcome to the Antichromatizer (v1.3.1)! *")
    print("********************************************\n")

#———————————————————————————————————————————————————————————————————————————————

#Prints footer
def outro():
    print("\n                                   \u2620*BOOM*\u2620")
    print("bye.\n\n")

#———————————————————————————————————————————————————————————————————————————————

#Gets valid pathname to an image file via file explorer window, opens it, calcs
#luminance values, and returns everything separately
def get_file():
    print(" *Who will be our victim today?")
    input("     (Press return to select your sacrifice)")
    root = tk.Tk()
    root.withdraw()
    img_path = fd.askopenfilename(title='FEED AN IMAGE INTO MY GAPING MAW')
    while (".png" not in (img_path.lower())) and \
          (".jpg" not in (img_path.lower())) and \
          (".jpeg" not in (img_path.lower())):
        print("\n Do not test me. Select a valid image file.")
        input("     (Press return to prove you are not as pathetic as this " + \
              "failure makes you seem)")
        img_path = fd.askopenfilename(title='FEED AN IMAGE INTO MY GAPING MAW')

    #Prevent file dialog from opening up again buggily
    root.destroy()
    
    #Open original file real quick
    op = img.open(img_path)
    op = (np.array(op))[:,:,0:3] #Discard transparency channel

    #Generate separate strings
    img_path = img_path.split("/")
    img_name = img_path[-1]
    filepath = ""
    for i in range(len(img_path)-1):
        filepath += (img_path[i] + "/")
    print("\n                 " + img_name + ", eh?")
    print("  ...Very well\n")

    #Luminance values will be needed
    lums = gen_lums(op)
    img_name = (img_name.split('.'))[0] #Toss file ext.
    
    return filepath,img_name,op,lums

#———————————————————————————————————————————————————————————————————————————————

#(Shifts, scales,) rounds, clips, converts to uint8
def to_uint(im_arr,scaling=True):
    if scaling and (np.min(im_arr) != np.max(im_arr)): #Avoid div0
        im_arr -= np.min(im_arr)
        im_arr = (im_arr * (255/np.max(im_arr)))
    im_arr = np.round(im_arr)
    im_arr = np.clip(im_arr,0,255)
    im_arr = im_arr.astype(np.uint8)
    return im_arr

#———————————————————————————————————————————————————————————————————————————————

def get_slice_n(lums):
    #Prompt for number of threshings
    print(" *How many Threshings shall we perform?")
    nt = -1
    while nt < 2 or nt > 512:
        nt = input("  (Must be on [2,512]. ~50 Recommended to start): ")
        if nt == '': nt = 50
        else:
            try: nt = int(nt)
            except ValueError: nt = -1

    #Threshing GIFspeed
    print("    **For how long shall we drag out these Threshings?")
    ts = -1
    while ts < 2 or ts > 250:
        ts = input("      (Must be on [2,250]. ~5 Recommended to start): ")
        if ts == '': ts = 5
        else:
            try: ts = int(ts)
            except ValueError: ts = -1

    #Prompt for number of shatters
    ns = -1
    print("\n *What number of Chrmoa-Breaks shall we perform?")
    while ns < 1 or ns > 255:
        ns = input("  (Must be on [1,255]. < 10 Recommended to start): ")
        if ns == '': ns = 10
        else:
            try: ns = int(ns)
            except ValueError: ns = -1

    #Threshing GIFspeed
    print("    **And each Shattering shall take...how long?")
    ss = -1
    while ss < 2 or ss > 250:
        ss = input("      (Must be on [2,250]. ~5 Recommended to start): ")
        if ss == '': ss = 5
        else:
            try: ss = int(ss)
            except ValueError: ss = -1

    #Reduce this shit
    num_lums = ((np.unique(lums)).shape)[0]
    num_lums_q = ((np.unique(to_uint(lums))).shape)[0]
    if ((num_lums) < nt) or ((num_lums_q) < ns):
        print("\n  Your image has been found to be deficient.")
        print("       Threshings reduced by: " + str(nt-num_lums))
        print("       Chroma-Breaks reduced by: " + str(ns-num_lums_q))
        nt = min(num_lums,nt)
        ns = min(num_lums_q,ns)
    
    return nt,ns,ts,ss

#———————————————————————————————————————————————————————————————————————————————

#Directs antichromatic shattering of the image
def shatter(filepath,img_name,lums,n_th,n_sh,t_th,t_sh):
    h,w = lums.shape
    
    #Init
    batch = np.empty((h,w,n_th+n_sh))
    
    #Threshold procedure for batch array generation
    batch[...,:n_th] = gen_threshed(lums,n_th,filepath,img_name)
    print("            ",end='')
    #Stepped procedure for batch array generation
    batch[...,n_th:] = gen_shattered(lums,n_sh,filepath,img_name)

    #Export
    return batch,batch_img_gen(batch,filepath,img_name,n_th,n_sh,t_th,t_sh)

#———————————————————————————————————————————————————————————————————————————————

#Generates euclidean luminance of each pixel in the image, returns 1-channel img
#However, ITU-R 601 7th Edition Construction of Luminance formula:
#L = R * 299/1000 + G * 587/1000 + B * 114/1000
#Or, Rec. 709 is:
#L = R * 2125/10000 + G * 7154/10000 + B * 0721/10000
#Might try one of those later
def gen_lums(op):
    h,w,c = op.shape
    lums = np.zeros((h,w),dtype=np.float32)
    #Calculate luminances of each pixel in the image
    for i in range(c):
        lums += np.square((op[...,i]).astype(np.float32))
    return np.sqrt(lums)
    
#———————————————————————————————————————————————————————————————————————————————

#Fast loop to generate thresholded images
#@jit(nopython=True)
#numba not used, so as to make output pretty
def gen_threshed(lums,slices,filepath,filename):
    #Init batch
    h,w = lums.shape
    batch = np.empty((h,w,slices))

    #Calculate boundaries
    dimmest = np.min(lums)
    brightest = np.max(lums)
    step = (brightest - dimmest) / (slices+1)
    
    perc = 0
    print("\u2620 \u2620 \u2620 ...",end='')
    #Test each threshold, quantize pixels over & under ——> 0 & 255
    for thresh in range(1,slices+1):
        #Subtract threshold_num * step_size from lums
        shard = lums - np.full((h,w), thresh*step)
        #Round to integers, clip to [0,1], and scale by 255
        shard = np.clip(np.round(shard),0,1) * 255
        #Put abs in the batch, in case of -0 elements
        batch[...,thresh-1] = np.abs(shard)
        #Print progress
        if thresh/slices >= perc:
            print("\u2620...",end='')
            perc += 1/10
    print(" \u2620 \u2620 \u2620")
    
    return batch

#———————————————————————————————————————————————————————————————————————————————

#Fast loop to generate stepped images
#@jit(nopython=True)
#numba not used, so as to make output pretty
def gen_shattered(lums,slices,filepath,filename):
    h,w = lums.shape
    #Init batch
    batch = np.zeros((h,w,slices))

    #Calculate boundaries
    dimmest = np.min(lums)
    brightest = np.max(lums)
    span = brightest - dimmest
    
    #Test each number of steps up to our max
    perc = 0
    print("\u2620 \u2620 \u2620 ...",end='')
    for n in range(slices):
        #Calculate lowest threshold
        low = dimmest + span/(n+2)
        #Generate all thresholds
        thresh = np.linspace(low,brightest,endpoint=False,num=n+1)
        #Bin the image according to these thresholds
        smashed = (np.digitize(lums,bins=thresh)).astype(np.float32)
        #Scale, but don't quantize or convert to uint
        if np.min(smashed) != np.max(smashed): smashed -= np.min(smashed)
        smashed *= (255.0/np.max(smashed))
        #Add to the batch
        batch[...,slices-1-n] = np.abs(smashed)
        if n/slices >= perc:
            print("\u2620...",end='')
            perc += 2/10
    print(" \u2620 \u2620 \u2620")

    return batch

#———————————————————————————————————————————————————————————————————————————————

#Fast loop to export the batch of images
#@jit(nopython=True)
#numba not used, many reasons
def batch_img_gen(batch,filepath,img_name,n_th,n_sh,t_th,t_sh):
    #Get sizes
    h,w,n = batch.shape
    assert n == n_th+n_sh
    filepath += img_name + "_ANTICHROMATIZATIONS_" + str(n_th) + "-" + \
                str(n_sh) + "/"
    os.makedirs(filepath, exist_ok=True)
    print("\n\n  *Exporting*\n  " + filepath,end='')
    print("\n  0%...",end='')
    
    #Holding each image from the batch and its inverse as PIL image objects
    shards = np.empty((n*2),dtype=type( \
             img.fromarray((batch[...,0]).astype(np.uint8), mode="L")))

    #Shatt imgs
    shatt_g = np.empty((n_sh * 4),dtype=type( \
             img.fromarray((batch[...,0]).astype(np.uint8), mode="L")))
    for i in range(n_sh):
        temp = to_uint(batch[...,i+n_th])
        #Normal img
        shards[i+2*n_th] = img.fromarray(temp, mode="L")
        shatt_g[i],shatt_g[-(i+1)] = shards[i+2*n_th],shards[i+2*n_th]
        #Inverted
        shards[i+n_th+n] = img.fromarray(255 - temp, mode="L")
        shatt_g[2*n_sh-1-i],shatt_g[2*n_sh+i] = shards[i+n_th+n],shards[i+n_th+n]
    print("5%...",end='')
    
    #Thresh images
    for i in range(n_th):
        temp = to_uint(batch[...,i])
        #Normal img
        shards[i] = img.fromarray(temp, mode="L")
        #Inverted
        shards[i+n_th] = img.fromarray(255 - temp, mode="L")
    print("15%...",end='')

    #Exporting: Thresh GIF
    header = "_" + img_name + ".gif"
    (shards[0]).save(filepath + "*" + "EMERGENT_" + str(n_th) + "-" + \
                str(t_th) + header, save_all = True, \
                append_images=shards[1:2*n_th], duration=t_th*10, \
                loop=0, mode="L")
    print("25%...",end='')
    
    #Exporting: Shatt GIF
    (shatt_g[0]).save(filepath + "*" + "HYPNOSIS_" + str(n_sh) + "-" + \
                 str(t_sh) + header, save_all = True, \
                 append_images=shatt_g[1:], duration=t_sh*10, \
                 loop=0, mode="L")
    print("30%...",end='')
    
    #Exporting: Thresh IMGs
    header = img_name + "_THRESH--"
    perc = 4/10
    for i in range(n_th):
        (shards[i]).save(filepath + header + str(i+1) + ".png", mode="L")
        (shards[i+n_th]).save(filepath + header[:-1] + str(i+1) + \
                        "_inv.png", mode="L")
        if i/n >= perc:
            print(format(perc*100,'.0f') + "%...",end='')
            perc += 1/10

    #Exporting: Shatt IMGs
    header = img_name + "_SHATT--"
    for i in range(n_sh):
        (shards[2*n_th+i]).save(filepath + header + str(n_sh-i) + \
                          ".png", mode="L")
        (shards[2*n-1-i]).save(filepath + header[:-1] + str(n_sh-i) + \
                         "_inv.png", mode="L")
        if (i+n_th)/n >= perc:
            print(format(perc*100,'.0f') + "%...",end='')
            perc += 1/10
    print("Done!")

    #Return it
    return filepath


#———————————————————————————————————————————————————————————————————————————————

#Calculates the per-pixel mean of a BW image batch array
def gen_mean(batch,scale=True,filepath=None,filename=None):
    #mean along the batch direction
    mean = np.mean(batch, axis=2)
    #Convert
    mean = to_uint(mean,scaling=scale)

    #Render
    if filepath is not None:
        outpath = filepath + "Combos/"
        os.makedirs(outpath, exist_ok=True)
        outpath += filename
        out = img.fromarray(mean, mode="L")
        out.save(outpath + "_MEAN.png", mode="L")
        out = img.fromarray(255 - mean, mode="L")
        out.save(outpath + "_xMEAN_inv.png", mode="L")

    return mean

#———————————————————————————————————————————————————————————————————————————————

#Calculates the per-pixel median of a BW image batch array
def gen_median(batch,scale=True,filepath=None,filename=None):
    #Calc median
    median = np.median(batch, axis=2)
    #Convert
    median = to_uint(median,scaling=scale)

    #Render
    if filepath is not None:
        outpath = filepath + "Combos/"
        os.makedirs(outpath, exist_ok=True)
        outpath += filename
        out = img.fromarray(median, mode="L")
        out.save(outpath + "_MEDIAN.png", mode="L")
        out = img.fromarray(255 - median, mode="L")
        out.save(outpath + "_xMEDIAN_inv.png", mode="L")

    return median

#———————————————————————————————————————————————————————————————————————————————

#Calculates the per-pixel modal average of a BW image batch array
def gen_mode(batch,scale=True,filepath=None,filename=None):
    #Fast-loop factor
    mode = make_it_count(batch)
    
    #Convert
    mode = to_uint(mode,scaling=scale)

    #Render
    if filepath is not None:
        outpath = filepath + "Combos/"
        os.makedirs(outpath, exist_ok=True)
        outpath += filename
        out = img.fromarray(mode, mode="L")
        out.save(outpath + "_MODE.png", mode="L")
        out = img.fromarray(255 - mode, mode="L")
        out.save(outpath + "_xMODE_inv.png", mode="L")

    return mode

#———————————————————————————————————————————————————————————————————————————————

#Fast loop for counting the mode--REALLY SLOW WITHOUT NUMBA
@jit(nopython=True)
def make_it_count(batch):
    #Round & get sizes
    d = 0
    batch = (np.round_(batch,d,batch)).astype(np.uint8)
    h,w,n = batch.shape

#Might be faster but optional args for np.unique() are not supportend in numba
##    #For each pixel, store the most commonly-ocurring unique element of batch
##    mode = np.empty((h,w),dtype=np.uint8)
##    for r in range(h):
##        for c in range(w):
##            pix_u, pix_i, pix_c = np.unique( batch[r,c,:] ,
##                                  return_inverse=True, return_counts=True)
##            mode[r,c] = pix_u[pix_i[np.argmax(pix_c)]]
##
##    return mode   
    
    #Count occurrences of each value for each pixel, through the batch
    counts = np.zeros((h,w,256),dtype=np.uint8)
    perc = h/4
    for row in range(h):
        if row > perc:
            perc += h/4
        for col in range(w):
            for i in range(n):
                counts[row,col,batch[row,col,i]] += 1

    #Return modal averages
    return (np.argmax(counts, axis=2))

#———————————————————————————————————————————————————————————————————————————————

#Generates image containing per-pixel outliers from a batch of BW images
def gen_outliers(batch,op,scale=True,filepath=None,filename=None):
    #Generate lum vals from original picture
    op_lums = gen_lums(op)
    h,w = op_lums.shape
    op_lums = np.reshape(op_lums,(h,w,1)) #reshape to have 'batch size' 1

    #Calculate pixelwise Luma difference from op for each image in batch
    lum_diffs = np.sqrt((op_lums - batch)**2)
    
    #Find indeces of biggest Outlier
    outlier_idx = np.argmax(lum_diffs,2)
    
    #Construct Outlier image from these indeces
    outliers = lie_out(h,w,batch,outlier_idx)
    
    #Convert
    outliers = to_uint(outliers,scaling=scale)

    #Render
    if filepath is not None:
        outpath = filepath + "Combos/"
        os.makedirs(outpath, exist_ok=True)
        outpath += filename
        out = img.fromarray(outliers, mode="L")
        out.save(outpath + "_OUTLIERS.png", mode="L")
        out = img.fromarray(255 - outliers, mode="L")
        out.save(outpath + "_xOUTLIERS_inv.png", mode="L")

    return outliers

#———————————————————————————————————————————————————————————————————————————————

#Fast loop for constructing the final image (not so slow without numba)
@jit(nopython=True)
def lie_out(h,w,batch,outlier_idx):
    outliers = np.empty((h,w),dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            outliers[r,c] = batch[r,c,outlier_idx[r,c]]
    return outliers

#———————————————————————————————————————————————————————————————————————————————
#Creates a plot with sliders to control the amt of each image.
#Not very well designed in terms of modularity. Also surprisingly slow. Oh well.
def disp_mixer(batch,names,outpath,filename):
    #——————————————————————————————
    #INIT
    
    h,w,n = batch.shape
    w_max = 1.0
    w_min = -w_max
    weights = np.zeros((n))

    #Generate image
    fig = plt.figure(figsize=(8.25,7.5))
    im_ax = plt.axes([0.1, 0.21, 0.8, 0.75])
    plt.axes(im_ax)
    plt.title('Conglomeration')
    im_ax.spines['top'].set_visible(False)
    im_ax.spines['left'].set_visible(False)
    im_ax.spines['bottom'].set_visible(False)
    im_ax.spines['right'].set_visible(False)
    im_ax.set_xticks([])
    im_ax.set_yticks([])
    im_plot = plt.imshow(weightem(batch,weights,2),cmap='gray',vmin=0,vmax=255)

    #Generate slider panels & put sliders innem
    slider_space = []
    sliders = []
    v_space = 0.185
    for y in range(n//2):
        slider_space.append(plt.axes([0.2, v_space - 0.02*(y), 0.2, 0.0125]))
    for y in range(n//2):
        slider_space.append(plt.axes([0.7, v_space - 0.02*(y), 0.2, 0.0125]))
    for snum, space in enumerate(slider_space):
        sliders.append(Slider(space, names[snum], w_min, w_max, \
                              valinit=weights[snum]))

    #Create buttons: Save, Shuffle, Zero
    savbutt = Button(plt.axes([0.5, 0.01, 0.1, 0.03]),"Export")
    savpath = outpath + "Combos/Conglomerations/"
    os.makedirs(savpath, exist_ok=True)
    savpath += filename + "_"
    shuffbutt = Button(plt.axes([0.75, 0.01, 0.1, 0.03]),"Shuffle")
    scentbutt = Button(plt.axes([0.25, 0.01, 0.1, 0.03]),"Zero")

    #——————————————————————————————
    #DEFINE WIDGET UPDATE FUNCTIONS

    #This function updates the image plot when the sliders are adjusted
    def update_weight(a):
        plt.axes(im_ax)
        #Get updated weigths data
        for i in range(n):
            weights[i] = sliders[i].val
        #Redraw plot
        im_plot = plt.imshow(weightem(batch,weights,2),cmap='gray', \
                  vmin=0,vmax=255)
        fig.canvas.draw_idle()

    #This function saves the currently plotted image to a subfolder
    def savout(a):
        for i in range(n):
            weights[i] = sliders[i].val
        
        imout = weightem(batch,weights,2)
        imout = img.fromarray(imout)
        imout.save(savpath + weight_str(weights) + "_CONGLOM.png", mode="L")

    #This function sets all sliders to random values
    def randval(a):
        weights = np.random.rand(n) * (w_max - w_min) + w_min
        for i in range(n):
            sliders[i].eventson = False
            sliders[i].set_val(weights[i])
            sliders[i].eventson = True
        update_weight(a)

    #This function sets all sliders to zero
    def zeroval(a):
        for i in range(n):
            sliders[i].eventson = False
            sliders[i].set_val(0)
            sliders[i].eventson = True
        update_weight(a)
            
    #——————————————————————————————
    #Specify what functions each widget runs
    for i in range(n):
        sliders[i].on_changed(update_weight)
        
    savbutt.on_clicked(savout)
    shuffbutt.on_clicked(randval)
    scentbutt.on_clicked(zeroval)

    #——————————————————————————————
    #Render the window
    plt.show()

#———————————————————————————————————————————————————————————————————————————————

#Gets a reasonable string of weights for exportation
def weight_str(weights):
    return '[' + ' '.join([format(n,'.4f') for n in weights]) + ']'

#———————————————————————————————————————————————————————————————————————————————

#Weighted average along axis provided
def weightem(batch,weights,axis):
    #Reshape
    nushape = (np.swapaxes(batch, batch.ndim-1, axis)).shape
    weights = np.broadcast_to(weights, nushape)
    weights = np.swapaxes(weights, batch.ndim-1, axis)
       
    #Find weighted average
    conglomeration = batch * weights
    conglomeration = np.sum(conglomeration,2)
    
    #Return as uint
    return to_uint(conglomeration)

#———————————————————————————————————————————————————————————————————————————————

#————————————#
# Run main() #
#————————————#

main()








