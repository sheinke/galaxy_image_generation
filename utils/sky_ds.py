import tensorflow as tf
import matplotlib.pyplot as plt

'''
used to load sky images based on google sky

to get a prepared subset of our images you can download them under

wget https://polybox.ethz.ch/index.php/s/hG38LeW0SKx1ZdK/download -o out.zip
unzip out.zip

Attention. Personal use only. Do NOT redistribute. 
'''


def get_img(path,nr_crops=2000):
    #read imge
    img = tf.io.read_file(path)

    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=1)
    
    #img = (tf.image.convert_image_dtype(img, tf.float32)/256)
    img = (tf.image.convert_image_dtype(img, tf.float32)/256-0.5)*2

    crops = [(tf.image.random_crop(img, size=[1024, 1024, 1]),-1) for _ in range(nr_crops)]

    

    return tf.data.Dataset.from_tensor_slices(crops)

def sky_ds(picture_path = f'./out/'):

    ds = tf.data.Dataset.list_files(f"{picture_path}*.png")

    #show one of the pictures used
    for batch in ds:
        print(batch)
        break

    #get images
    ds = ds.flat_map(get_img)

    return ds

ds = sky_ds()

#print one of images
'''
for batch in ds:
    print(batch.shape)
    plt.imshow(batch[:,:,0],cmap='gray')
    plt.show()
    break
'''