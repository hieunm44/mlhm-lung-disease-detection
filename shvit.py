import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Input, ReLU, LayerNormalization, Identity
from tensorflow.keras.models import Model, Sequential, load_model

@register_keras_serializable()
class Conv2D_BN(keras.layers.Layer):
    def __init__(self, filters, kernel_size=1, strides=1, padding='same'):
        super().__init__()
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn = BatchNormalization()

    def call(self, inputs):
        x = self.conv(inputs)

        return self.bn(x)


@register_keras_serializable()
class PatchMerging(keras.layers.Layer):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.conv1 = Conv2D_BN(filters=dim*4, kernel_size=1, strides=1)
        self.act = ReLU()
        self.conv2 = Conv2D_BN(filters=dim*4, kernel_size=3, strides=2, padding='same')
        self.conv3 = Conv2D_BN(filters=output_dim, kernel_size=1, strides=1, padding='valid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        return x


@register_keras_serializable()
class Residual(keras.layers.Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def call(self, inputs):
        x = inputs + self.layer(inputs)

        return x


@register_keras_serializable()
class FFN(keras.layers.Layer):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.pw1 = Conv2D_BN(filters=hidden_dim)
        self.act = ReLU()
        self.pw2 = Conv2D_BN(filters=embed_dim)

    def call(self, inputs):
        x = self.pw1(inputs)
        x = self.act(x)
        x = self.pw2(x)

        return x


@register_keras_serializable()
class SHSA(keras.layers.Layer):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = LayerNormalization()
        self.qkv = Conv2D_BN(filters=qk_dim*2 + pdim, kernel_size=1)
        self.proj = Sequential([
            ReLU(),
            Conv2D_BN(filters=dim)
        ])

    def call(self, inputs): # self.blocks2, BasicBlock 1, dim=128, qk_dim=16, pdim=64, inputs.shape=(2, 2, 224)
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3] # B=128, H=2, W=2, C=224

        # Split into two parts
        x1, x2 = tf.split(inputs, [self.pdim, self.dim - self.pdim], axis=-1) # (2, 2, 48), (2, 2, 176)

        # Normalize x1
        x1 = self.pre_norm(x1) # (2, 2, 48)

        # Compute Q, K, V
        qkv = self.qkv(x1) # (2, 2, 80)
        q, k, v = tf.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], axis=-1) # (2, 2, 16) (2, 2, 16) (2, 2, 48)

        # Reshape for attention computation
        q = tf.reshape(q, [B, H*W, self.qk_dim]) # (4, 16)
        k = tf.reshape(k, [B, H*W, self.qk_dim]) # (4, 16)
        v = tf.reshape(v, [B, H*W, self.pdim])   # (4, 48)

        # Compute attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale # (4, 4)
        attn = tf.nn.softmax(attn, axis=-1) # (4, 4)
        attn = tf.matmul(attn, v) # (4, 48)

        x1 = tf.reshape(attn, [B, H, W, self.pdim]) # (2, 2, 48)

        # Concatenate with x2 and project
        x = tf.concat([x1, x2], axis=-1) # (2, 2, 224)
        x = self.proj(x) # (2, 2, 224)


        return x


@register_keras_serializable()
class BasicBlock(keras.layers.Layer):
    def __init__(self, dim, qk_dim, pdim, block_type="s"):
        super().__init__()
        self.conv = Residual(Conv2D_BN(dim, 3, padding="same"))

        if block_type=='s':  # Later stages
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        elif block_type=='i':  # Early stages
            self.mixer = Identity()

        self.ffn = Residual(FFN(dim, dim * 2))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.mixer(x)
        x = self.ffn(x)

        return x


@register_keras_serializable()
class SHVIT(keras.Model):
    def __init__(self,
                 embed_dim=[128, 256, 384],
                 qk_dim=[16, 16, 16],
                 partial_dim=[32, 64, 96],
                 depth=[1, 2, 3],
                 down_ops=['subsample', 'subsample', ''],
                 types=['s', 's', 's'],
                 num_classes=14,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_embed = Sequential([
            Conv2D_BN(filters=embed_dim[0]//8, kernel_size=3, strides=2, padding='same'), ReLU(), # filters=16
            Conv2D_BN(filters=embed_dim[0]//4, kernel_size=3, strides=2, padding='same'), ReLU(), # filters=32
            Conv2D_BN(filters=embed_dim[0]//2, kernel_size=3, strides=2, padding='same'), ReLU(), # filters=64
            Conv2D_BN(filters=embed_dim[0], kernel_size=3, strides=2, padding='same') # filters=128
        ])

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        for i, (ed, kd, pd, dpth, do, t) in enumerate(zip(embed_dim, qk_dim, partial_dim, depth, down_ops, types)): # embed_dim=[128, 224, 320], depth=[2, 4, 5], partial_dim=[32, 48, 68], types=['i', 's', 's']
            for _ in range(dpth):
                eval('self.blocks'+str(i+1)).append(BasicBlock(dim=ed, qk_dim=kd, pdim=pd, block_type=t))
            if do == 'subsample':
                blk = eval('self.blocks'+str(i+2))
                blk.append(Sequential([
                    Residual(Conv2D_BN(filters=ed, kernel_size=3, strides=1, padding='same')),
                    Residual(FFN(embed_dim=ed, hidden_dim=ed*2))
                    ]))
                blk.append(PatchMerging(dim=ed, output_dim=embed_dim[i+1]))
                blk.append(Sequential([
                    Residual(Conv2D_BN(filters=embed_dim[i+1], kernel_size=3, strides=1, padding='same')),
                    Residual(FFN(embed_dim=embed_dim[i+1], hidden_dim=embed_dim[i+1]*2))
                    ]))

        self.blocks1 = Sequential([*self.blocks1]) # i=0: 2 BasicBlocks
        self.blocks2 = Sequential([*self.blocks2]) # i=0: Conv2D_BN -> FFN -> PatchMerging -> Conv2D_BN -> FFN; i=1: 4 BasicBlocks
        self.blocks3 = Sequential([*self.blocks3]) # i=1: Conv2D_BN -> FFN -> PatchMerging -> Conv2D_BN -> FFN; i=2: 5 BasicBlocks

        self.pool = GlobalAveragePooling2D()
        self.fc = Sequential([
            Dense(128, activation='relu'),
            Dense(num_classes, activation='sigmoid')
            ])

    def call(self, inputs, training=True): # (64, 64, 1)
        x = self.patch_embed(inputs) # (32, 32, 16) -> (16, 16, 32) -> (8, 8, 64) -> (4, 4, 128)
        x = self.blocks1(x) # BasicBlock 1: (4, 4, 128) -> (4, 4, 128) -> (4, 4, 128); BasicBlock 2: (4, 4, 128) -> (4, 4, 128) -> (4, 4, 128)
        x = self.blocks2(x) # Conv2D_BN: (4, 4, 128) -> FFN: (4, 4, 128) -> PatchMerging: (2, 2, 224) -> Conv2D_BN: (2, 2, 224) -> FFN: (2, 2, 224) -> BasicBlock 1->4: (2, 2, 224)
        x = self.blocks3(x) # Conv2D_BN: (2, 2, 224) -> FFN: (2, 2, 224) -> PatchMerging: (1, 1, 320) -> Conv2D_BN: (1, 1, 320) -> FFN: (1, 1, 320) -> BasicBlock 1->5: (1, 1, 320)
        x = self.pool(x) # (320)
        x = self.fc(x) # (14)

        return x