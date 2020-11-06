from PIL import Image
from math import floor, log2
import numpy as np
import time
from random import random
import os
import wandb
import sys
import argparse

sys.path.append("..")

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K
import tensorflow as tf

from conv_mod import *
from utils.data_utils import load_data, DataUtilsConfig

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run_name', type=str, required=True, help='model version')
parser.add_argument('--data_root', type=str, default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/',
                    help='path to the dataset')
parser.add_argument('--use_scratch', type=int, default=1, help='set to 0 if results not to be put on Leonhard scratch')

parser.add_argument('--run_name_of_loaded_model', type=str, default='', help='run name of model to be loaded')
parser.add_argument('--number_of_loaded_model', type=int, default=1, help='number of checkpointed model to be loaded')
parser.add_argument('--set_steps', type=int, default=0, help='set steps to nr of model * 10000')

parser.add_argument('--evaluate_mode', type=int, default=0, help='non zero if no training but evaluation required')
parser.add_argument('--nr_eval_images', type=int, default=1, help='number of evaluation images to be generated')

parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--debug_mode', type=int, default=0, help='1 if in debug mode')
parser.add_argument('--disc_update_prob', type=float, default=0.9, help='discriminator update probability')
parser.add_argument('--mixed_prob', type=float, default=0.9, help='alternating training probability')
parser.add_argument('--im_size', type=int, default=1024, help='image size, power of 2')
parser.add_argument('--latent_size', type=int, default=128, help='latent dimensionality')
parser.add_argument('--cha', type=int, default=8, help='number of channels')
parser.add_argument('--img_subtract_mean', type=int, default=0, help='subtract mean from images')
parser.add_argument('--img_binarise', type=int, default=0, help='binarise images')
parser.add_argument('--sky_ds', type=int, default=0, help='use sky images')

opt = parser.parse_args()

# setup wandb for visualization
wandb_run = True
if wandb_run:
    wandb.init(project="cil_stylegan", sync_tensorboard=True)
    wandb.config.update(opt)

# set result paths
scratch_path = "."
if (not opt.debug_mode):
    scratch_path = os.environ["SCRATCH"]
run_name = opt.run_name
MODEL_PATH = scratch_path + f"/{run_name}/Model/"
RESULT_PATH = scratch_path + f"/{run_name}/Results/"

if len(opt.run_name_of_loaded_model):
    LOAD_PATH = scratch_path + f"/{opt.run_name_of_loaded_model}/Model/"

if not opt.evaluate_mode:
    os.makedirs(MODEL_PATH,exist_ok = True)
    os.makedirs(RESULT_PATH,exist_ok = True)

# image size changed to be power of 2
im_size = opt.im_size
# latent_size can be changed. it is later processed by a net to produce a latent size of 512 for each of the n_layer
latent_size = opt.latent_size
# reduced channels to 8 from 24 original because we only use grayscale
cha = opt.cha
# batch size
BATCH_SIZE = opt.batch_size
if opt.debug_mode:
    BATCH_SIZE = 1

n_layers = int(log2(im_size) - 1)

mixed_prob = opt.mixed_prob
disc_prob = opt.disc_update_prob

DataConfig = DataUtilsConfig()
DataConfig.data_root = scratch_path + opt.data_root
DataConfig.img_size=opt.im_size
DataConfig.img_subtract_mean = opt.img_subtract_mean
DataConfig.img_binarise = opt.img_binarise
DataConfig.batch_size = BATCH_SIZE
DataConfig.resize_method = 'padded'
DataConfig.pad_size = 100 - 24
DataConfig.score_for_actual_labeled = 1.25
DataConfig.sky_ds = opt.sky_ds

def noise(n):
    return np.random.normal(0.0, 1.0, size=[n, latent_size]).astype('float32')


def noiseList(n):
    return [noise(n)] * n_layers


def mixedList(n):
    tt = int(random() * n_layers)
    p1 = [noise(n)] * tt
    p2 = [noise(n)] * (n_layers - tt)
    return p1 + [] + p2


def nImage(n):
    return np.random.uniform(0.0, 1.0, size=[n, im_size, im_size, 1]).astype('float32')


# Loss functions
def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                             axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * weight


def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))


def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# Lambdas
def crop_to_fit(x):
    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]


def upsample(x):
    return K.resize_images(x, 2, 2, "channels_last", interpolation='bilinear')


def upsample_to_size(x):
    y = im_size // x.shape[2]
    x = K.resize_images(x, y, y, "channels_last", interpolation='bilinear')
    return x


# Blocks
def g_block(inp, istyle, inoise, fil, u=True):
    if u:
        # Custom upsampling because of clone_model issue
        out = Lambda(upsample, output_shape=[None, inp.shape[2] * 2, inp.shape[2] * 2, None])(inp)
    else:
        out = Activation('linear')(inp)

    rgb_style = Dense(fil, kernel_initializer=VarianceScaling(200 / out.shape[2]))(istyle)
    style = Dense(inp.shape[-1], kernel_initializer='he_uniform')(istyle)
    delta = Lambda(crop_to_fit)([inoise, out])
    d = Dense(fil, kernel_initializer='zeros')(delta)

    out = Conv2DMod(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
    out = add([out, d])
    out = LeakyReLU(0.2)(out)

    style = Dense(fil, kernel_initializer='he_uniform')(istyle)
    d = Dense(fil, kernel_initializer='zeros')(delta)

    out = Conv2DMod(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
    out = add([out, d])
    out = LeakyReLU(0.2)(out)

    return out, to_rgb(out, rgb_style)


def d_block(inp, fil, p=True):
    res = Conv2D(fil, 1, kernel_initializer='he_uniform')(inp)

    out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inp)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')(out)
    out = LeakyReLU(0.2)(out)

    out = add([res, out])

    if p:
        out = AveragePooling2D()(out)

    return out


def to_rgb(inp, style):
    size = inp.shape[2]
    # this layer produced actual rgb color but we now have a mapping to a single grayscale
    x = Conv2DMod(1, 1, kernel_initializer=VarianceScaling(200 / size), demod=False)([inp, style])
    return Lambda(upsample_to_size, output_shape=[None, im_size, im_size, None])(x)


def from_rgb(inp, conc=None):
    # not used
    fil = int(im_size * 4 / inp.shape[2])
    z = AveragePooling2D()(inp)
    x = Conv2D(fil, 1, kernel_initializer='he_uniform')(z)
    if conc is not None:
        x = concatenate([x, conc])
    return x, z


class GAN(object):

    def __init__(self, steps=1, lr=0.0001, decay=0.00001):

        # Models
        self.D = None
        self.S = None
        self.G = None

        self.GE = None
        self.SE = None

        self.DM = None
        self.AM = None

        # Config
        self.LR = lr
        self.steps = steps
        self.beta = 0.999

        # Init Models
        self.discriminator()
        self.generator()

        self.GMO = Adam(lr=self.LR, beta_1=0, beta_2=0.999)
        self.DMO = Adam(lr=self.LR, beta_1=0, beta_2=0.999)

        self.GE = clone_model(self.G)
        self.GE.set_weights(self.G.get_weights())

        self.SE = clone_model(self.S)
        self.SE.set_weights(self.S.get_weights())

    def discriminator(self):

        if self.D:
            return self.D

        # input is now only one color channel
        inp = Input(shape=[im_size, im_size, 1])

        # because of the reduced channel size overall information per layer stays the same
        x = d_block(inp, 1 * cha)  # 128

        x = d_block(x, 2 * cha)  # 64

        x = d_block(x, 4 * cha)  # 32

        x = d_block(x, 6 * cha)  # 16

        x = d_block(x, 8 * cha)  # 8

        x = d_block(x, 16 * cha)  # 4

        x = d_block(x, 32 * cha, p=False)  # 4

        x = Flatten()(x)

        x = Dense(1, kernel_initializer='he_uniform', name='dis_out')(x)

        self.D = Model(inputs=inp, outputs=x)

        return self.D

    def generator(self):

        if self.G:
            return self.G

        # === Style Mapping ===
        # network mapping latent space to latent space for each layer
        self.S = Sequential()

        self.S.add(Dense(512, input_shape=[latent_size]))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))

        # === Generator ===

        # Inputs
        inp_style = []

        for i in range(n_layers):
            inp_style.append(Input([512]))

        inp_noise = Input([im_size, im_size, 1])

        # Latent
        x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])

        outs = []

        # Actual Model
        x = Dense(4 * 4 * 4 * cha, activation='relu', kernel_initializer='random_normal')(x)
        x = Reshape([4, 4, 4 * cha])(x)

        x, r = g_block(x, inp_style[0], inp_noise, 32 * cha, u=False)  # 4
        outs.append(r)

        x, r = g_block(x, inp_style[1], inp_noise, 16 * cha)  # 8
        outs.append(r)

        x, r = g_block(x, inp_style[2], inp_noise, 8 * cha)  # 16
        outs.append(r)

        x, r = g_block(x, inp_style[3], inp_noise, 6 * cha)  # 32
        outs.append(r)

        x, r = g_block(x, inp_style[4], inp_noise, 4 * cha)  # 64
        outs.append(r)

        x, r = g_block(x, inp_style[5], inp_noise, 2 * cha)  # 128
        outs.append(r)

        x, r = g_block(x, inp_style[6], inp_noise, 1 * cha)  # 256
        outs.append(r)

        x = add(outs)

        x = Lambda(lambda y: y / 2 + 0.5, name='gen_out')(
            x)  # Use values centered around 0, but normalize to [0, 1], providing better initialization

        self.G = Model(inputs=inp_style + [inp_noise], outputs=x)

        return self.G

    def GenModel(self):

        # Generator Model for Evaluation

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.S(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])

        gf = self.G(style + [inp_noise])

        self.GM = Model(inputs=inp_style + [inp_noise], outputs=gf)

        return self.GM

    def GenModelA(self):

        # Parameter Averaged Generator Model

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.SE(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])

        gf = self.GE(style + [inp_noise])

        self.GMA = Model(inputs=inp_style + [inp_noise], outputs=gf)

        return self.GMA

    def EMA(self):

        # Parameter Averaging

        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1 - self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1 - self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    def MAinit(self):
        # Reset Parameter Averaging
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())


class StyleGAN(object):

    def __init__(self, steps=1, lr=0.0001, decay=0.00001, silent=True):

        # Init GAN and Eval Models
        self.GAN = GAN(steps=steps, lr=lr, decay=decay)
        self.GAN.GenModel()  # self.GM
        self.GAN.GenModelA()

        self.GAN.G.summary()
        self.GAN.D.summary()
        # self.GAN.S.summary()

        # Data generator to get galaxies
        if not opt.evaluate_mode:
            if opt.debug_mode:
                self.ds = load_data(DataConfig).take(4)
            else:
                self.ds = load_data(DataConfig)

        # Set up variables
        self.lastblip = time.clock()

        self.silent = silent

        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

        self.pl_mean = 0
        self.av = np.zeros([44])

    def train(self):
        for epoch in range(100):
            print('#####')
            print(f'epoch {epoch + 1}')
            for image_batch, labels in self.ds:
                # Train Alternating
                if random() < mixed_prob:
                    style = mixedList(BATCH_SIZE)
                else:
                    style = noiseList(BATCH_SIZE)

                # Apply penalties every 16 steps
                apply_gradient_penalty = self.GAN.steps % 2 == 0 or self.GAN.steps < 10000
                apply_path_penalty = self.GAN.steps % 16 == 0

                a, b, c, d = self.train_step(image_batch, style, nImage(BATCH_SIZE), apply_gradient_penalty,
                                             apply_path_penalty)

                # Adjust path length penalty mean
                # d = pl_mean when no penalty is applied
                if self.pl_mean == 0:
                    self.pl_mean = np.mean(d)
                self.pl_mean = 0.99 * self.pl_mean + 0.01 * np.mean(d)

                if self.GAN.steps % 10 == 0 and self.GAN.steps > 20000:
                    self.GAN.EMA()

                if self.GAN.steps <= 25000 and self.GAN.steps % 1000 == 2:
                    self.GAN.MAinit()

                if np.isnan(a):
                    print("NaN Value Error.")
                    exit()

                # Print info
                if opt.debug_mode or (self.GAN.steps % 100 == 0 and not self.silent):
                    print("\n\nRound " + str(self.GAN.steps) + ":")
                    print("D:", np.array(a))
                    print("G:", np.array(b))
                    print("PL:", self.pl_mean)
                    if wandb_run:
                        wandb.log({'D loss': np.array(a)})
                        wandb.log({'G loss': np.array(b)})

                    s = round((time.clock() - self.lastblip), 4)
                    self.lastblip = time.clock()

                    steps_per_second = 100 / s
                    steps_per_minute = steps_per_second * 60
                    steps_per_hour = steps_per_minute * 60
                    print("Steps/Second: " + str(round(steps_per_second, 2)))
                    print("Steps/Hour: " + str(round(steps_per_hour)))

                    min1k = floor(1000 / steps_per_minute)
                    sec1k = floor(1000 / steps_per_second) % 60
                    print("1k Steps: " + str(min1k) + ":" + str(sec1k))
                    steps_left = 200000 - self.GAN.steps + 1e-7
                    hours_left = steps_left // steps_per_hour
                    minutes_left = (steps_left // steps_per_minute) % 60

                    print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
                    print()

                    # Save Model
                    if self.GAN.steps % 500 == 0 or opt.debug_mode:
                        self.save(floor(self.GAN.steps / 10000))
                    if self.GAN.steps % 1000 == 0 or (self.GAN.steps % 100 == 0 and self.GAN.steps < 2500):
                        self.evaluate(floor(self.GAN.steps / 1000))

                self.GAN.steps = self.GAN.steps + 1

    @tf.function
    def train_step(self, images, style, noise, perform_gp=True, perform_pl=False):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Get style information
            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(style)):
                w_space.append(self.GAN.S(style[i]))

            # Generate images
            generated_images = self.GAN.G(w_space + [noise])

            # Discriminate
            real_output = self.GAN.D(images, training=True)
            fake_output = self.GAN.D(generated_images, training=True)

            # Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            if perform_gp:
                # R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                # Slightly adjust W space
                w_space_2 = []
                for i in range(len(style)):
                    std = 0.1 / (K.std(w_space[i], axis=0, keepdims=True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                # Generate from slightly adjusted W space
                pl_images = self.GAN.G(w_space_2 + [noise])

                # Get distance after adjustment (path length)
                delta_g = K.mean(K.square(pl_images - generated_images), axis=[1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        # Get gradients for respective areas
        gradients_of_generator = gen_tape.gradient(gen_loss, self.GAN.GM.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.GAN.D.trainable_variables)

        # Apply gradients
        self.GAN.GMO.apply_gradients(zip(gradients_of_generator, self.GAN.GM.trainable_variables))
        if random() < disc_prob:
            self.GAN.DMO.apply_gradients(zip(gradients_of_discriminator, self.GAN.D.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths

    def evaluate(self, num=0, trunc=1.0):

        n1 = noiseList(64)
        n2 = nImage(64)
        trunc = np.ones([64, 1]) * trunc

        generated_images = self.GAN.GM.predict(n1 + [n2], batch_size=BATCH_SIZE)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i + 8], axis=1))

        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)
        print(f'shape: {np.uint8(c1 * 255)[:, :, 0].shape}')
        x = Image.fromarray(np.uint8(c1 * 255)[:, :, 0], mode='L')
        
        if wandb_run:
            wandb.log({"generated": [wandb.Image(x, caption="galaxies_step_" + str(self.GAN.steps))]})

        x.save(RESULT_PATH + "i" + str(num) + ".png")

        # Moving Average
        # the next two are not very usefull to us, but i left them in case the results are good

        generated_images = self.GAN.GMA.predict(n1 + [n2, trunc], batch_size=BATCH_SIZE)
        # generated_images = self.generateTruncated(n1, trunc = trunc)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i + 8], axis=1))

        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1 * 255)[:, :, 0], mode='L')

        x.save(RESULT_PATH + "i" + str(num) + "-ema.png")

        # Mixing Regularities
        nn = noise(8)
        n1 = np.tile(nn, (8, 1))
        n2 = np.repeat(nn, 8, axis=0)
        tt = int(n_layers / 2)

        p1 = [n1] * tt
        p2 = [n2] * (n_layers - tt)

        latent = p1 + [] + p2

        generated_images = self.GAN.GMA.predict(latent + [nImage(64), trunc], batch_size=BATCH_SIZE)
        # generated_images = self.generateTruncated(latent, trunc = trunc)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i + 8], axis=0))

        c1 = np.concatenate(r, axis=1)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1 * 255)[:, :, 0], mode='L')

        x.save(RESULT_PATH + "i" + str(num) + "-mr.png")

    def generateTruncated(self, style, noi=np.zeros([44]), trunc=0.5, outImage=False, num=0):
        # not used
        # Get W's center of mass
        if self.av.shape[0] == 44:  # 44 is an arbitrary value
            print("Approximating W center of mass")
            self.av = np.mean(self.GAN.S.predict(noise(2000), batch_size=64), axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        if noi.shape[0] == 44:
            noi = nImage(64)

        w_space = []
        pl_lengths = self.pl_mean
        for i in range(len(style)):
            tempStyle = self.GAN.S.predict(style[i])
            tempStyle = trunc * (tempStyle - self.av) + self.av
            w_space.append(tempStyle)

        generated_images = self.GAN.GE.predict(w_space + [noi], batch_size=BATCH_SIZE)

        if outImage:
            r = []

            for i in range(0, 64, 8):
                r.append(np.concatenate(generated_images[i:i + 8], axis=0))

            c1 = np.concatenate(r, axis=1)
            c1 = np.clip(c1, 0.0, 1.0)

            x = Image.fromarray(np.uint8(c1 * 255))

            x.save(RESULT_PATH + "t" + str(num) + ".png")

        return generated_images

    def saveModel(self, model, name, num):
        json = model.to_json()
        with open(MODEL_PATH + name + ".json", "w") as json_file:
            json_file.write(json)

        model.save_weights(MODEL_PATH + name + "_" + str(num) + ".h5")

    def loadModel(self, name, num):

        print("***** loading model {} from {}".format(LOAD_PATH, num))
        file = open(LOAD_PATH + name + ".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json, custom_objects={'Conv2DMod': Conv2DMod})
        mod.load_weights(LOAD_PATH + name + "_" + str(num) + ".h5")

        return mod

    def loadModel_extern(self, name, num, LOAD_PATH):
        # those parser flags are nice. To bad its harder to call methods now
        print("***** loading model {} from {}".format(LOAD_PATH, num))
        file = open(LOAD_PATH + name + ".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json, custom_objects={'Conv2DMod': Conv2DMod})
        mod.load_weights(LOAD_PATH + name + "_" + str(num) + ".h5")

        return mod

    def load_extern(self, num, LOAD_PATH):  # Load JSON and Weights from /Models/

        # Load Models
        self.GAN.D = self.loadModel_extern("dis", num, LOAD_PATH)
        self.GAN.S = self.loadModel_extern("sty", num, LOAD_PATH)
        self.GAN.G = self.loadModel_extern("gen", num, LOAD_PATH)

        self.GAN.GE = self.loadModel_extern("genMA", num, LOAD_PATH)
        self.GAN.SE = self.loadModel_extern("styMA", num, LOAD_PATH)

        self.GAN.GenModel()
        self.GAN.GenModelA()

    def save(self, num):  # Save JSON and Weights into /Models/
        self.saveModel(self.GAN.S, "sty", num)
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)

        self.saveModel(self.GAN.GE, "genMA", num)
        self.saveModel(self.GAN.SE, "styMA", num)

    def load(self, num):  # Load JSON and Weights from /Models/

        # Load Models
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.S = self.loadModel("sty", num)
        self.GAN.G = self.loadModel("gen", num)

        self.GAN.GE = self.loadModel("genMA", num)
        self.GAN.SE = self.loadModel("styMA", num)

        self.GAN.GenModel()
        self.GAN.GenModelA()

    def gen_samples(self):
        #prediction method. so far without flags
        scratch_path = os.environ["SCRATCH"]
        path = f'{scratch_path}/finetuned/Model/'

        model.load_extern(47, path)

        print('model loaded')

        for i in range(1000):
            print(i)
            n1 = noiseList(64)
            n2 = nImage(64)

            generated_images = model.GAN.GM.predict(n1 + [n2], batch_size=8)

            r = []

            for j in range(0, 64, 8):
                r.append(np.concatenate(generated_images[j:j + 8], axis=1))

            c1 = np.concatenate(r, axis=0)
            c1 = np.clip(c1, 0.0, 1.0)

            c1 = c1 [12:1012,12:1012,:]

            print(f'shape: {np.uint8(c1 * 255)[:, :, 0].shape}')
            x = Image.fromarray(np.uint8(c1 * 255)[:, :, 0], mode='L')
            
            x.save(f'{scratch_path}/finetuned/Samples/' +  str(i) + ".png",optimize=False, compress_level=0)

        exit()

if __name__ == "__main__":
    model = StyleGAN(lr=0.0001, silent=False)

    if opt.run_name_of_loaded_model:
        model.load(opt.number_of_loaded_model)
        if opt.set_steps:
            model.steps = opt.number_of_loaded_model * 10000

    if opt.evaluate_mode and not opt.run_name_of_loaded_model:
        print("Please specify model for evaluation!")
        quit()

    if False:
        model.gen_samples()

    if opt.evaluate_mode:
        RESULT_PATH = scratch_path + f"/{opt.run_name}/Evaluation/"
        os.makedirs(RESULT_PATH,exist_ok = True)
        for i in range(0, opt.nr_eval_images):
            model.evaluate(i)
    else:
        model.train()
