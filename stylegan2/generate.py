import numpy as np 
import imageio
import os
from PIL import Image
import tensorflow as tf

def gen_video_pretrain():
    scratch_path = os.environ["SCRATCH"]

    max_len = 598
    print(max_len)
    print(f"{scratch_path}/pretraining_long2/Results/i{0}.png")
    im = np.array(Image.open(f"{scratch_path}/pretraining_long2/Results/i{0}.png"))
    print(im.shape)

    dim = 3
    seq = np.zeros((max_len,1000,2000),dtype=np.uint8)

    
    for i in range(max_len):
        print(i)
        vid = np.array(Image.open(f"{scratch_path}/pretraining_long2/Results/i{i}.png"),dtype=np.uint8)

        #print((i//5)*80,(i//5)*80+80,vid.shape)
        seq[i,:,0:1000]=vid[12:1012,12:1012]
        seq[i,:,1000:2000]=vid[12:1012,1024+12:1024+1012]

    #plt.imshow(seq[0])
    imageio.mimsave(f'{scratch_path}/video/pretraining.gif', seq)

def gen_video_finetune():
    scratch_path = os.environ["SCRATCH"]

    max_len = 480 #598
    print(max_len)
    print(f"{scratch_path}/finetuning2/Results/i{0}.png")
    im = np.array(Image.open(f"{scratch_path}/finetuning2/Results/i{0}.png"))
    print(im.shape)

    dim = 3
    seq = np.zeros((max_len,1000,2000),dtype=np.uint8)

    
    for i in range(max_len):
        print(i)
        vid = np.array(Image.open(f"{scratch_path}/finetuning2/Results/i{i}.png"),dtype=np.uint8)

        #print((i//5)*80,(i//5)*80+80,vid.shape)
        seq[i,:,0:1000]=vid[12:1012,12:1012]
        seq[i,:,1000:2000]=vid[12:1012,1000+12:1000+1012]

    #plt.imshow(seq[0])
    imageio.mimsave(f'{scratch_path}/video/finetuning.gif', seq)


def get_image():
    scratch_path = os.environ["SCRATCH"]

    vid = np.array(Image.open(f"{scratch_path}/pretraining_long2/Results/i{597}.png"),dtype=np.uint8)
    vid = vid[12:1012,12:1012]

    x = Image.fromarray(vid, mode='L')
            
    x.save(f'{scratch_path}/video/pretraining.png',optimize=False, compress_level=0)

    vid = np.array(Image.open(f"{scratch_path}/finetuning2/Results/i{476}.png"),dtype=np.uint8)
    vid = vid[12:1012,1000+12:1000+1012]#vid[12:1012,12:1012]

    x = Image.fromarray(vid, mode='L')
            
    x.save(f'{scratch_path}/video/finetuning.png',optimize=False, compress_level=0)


if __name__ == "__main__":
    #gen_video_pretrain()
    #gen_video_finetune()
    #gen_samples()

    get_image()