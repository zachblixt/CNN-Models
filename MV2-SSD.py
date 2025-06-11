import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# L2 normalization for conv4_3
class L2Normalization(layers.Layer):
    def __init__(self, gamma=20, epsilon=1e-10):
        super(L2Normalization, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma_tensor = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.gamma),
            trainable=True,
            name='gamma'
        )

    def call(self, x):
        norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        x = x / norm
        return x * self.gamma_tensor


class SSD(tf.keras.Model):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # Base VGG16 until conv4_3
        vgg = VGG16(weights='imagenet', include_top=False)
        self.features = models.Sequential(vgg.layers[:15])  # Up to conv4_3
        self.l2norm = L2Normalization()

        # Additional VGG layers (conv5 block is skipped; SSD uses conv6/conv7)
        self.conv6 = layers.Conv2D(1024, kernel_size=3, padding='same', dilation_rate=6, activation='relu')  # fc6
        self.conv7 = layers.Conv2D(1024, kernel_size=1, activation='relu')  # fc7

        # Extra SSD layers
        self.extras = [
            layers.Conv2D(256, 1, activation='relu'),  # conv8_1
            layers.Conv2D(512, 3, strides=2, padding='same', activation='relu'),  # conv8_2
            layers.Conv2D(128, 1, activation='relu'),  # conv9_1
            layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),  # conv9_2
            layers.Conv2D(128, 1, activation='relu'),  # conv10_1
            layers.Conv2D(256, 3, activation='relu'),  # conv10_2
            layers.Conv2D(128, 1, activation='relu'),  # conv11_1
            layers.Conv2D(256, 3, activation='relu')   # conv11_2
        ]

        # Localization and confidence prediction layers
        self.loc = [
            layers.Conv2D(4 * 4, kernel_size=3, padding='same'),  # conv4_3: 4 boxes
            layers.Conv2D(6 * 4, kernel_size=3, padding='same'),  # conv7: 6 boxes
            layers.Conv2D(6 * 4, kernel_size=3, padding='same'),  # conv8_2
            layers.Conv2D(6 * 4, kernel_size=3, padding='same'),  # conv9_2
            layers.Conv2D(4 * 4, kernel_size=3, padding='same'),  # conv10_2
            layers.Conv2D(4 * 4, kernel_size=3, padding='same')   # conv11_2
        ]

        self.conf = [
            layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # conv4_3
            layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # conv7
            layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # conv8_2
            layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # conv9_2
            layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # conv10_2
            layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')   # conv11_2
        ]

    def call(self, x):
        locs = []
        confs = []

        # conv4_3
        conv4_3_feats = self.features(x)
        norm_feats = self.l2norm(conv4_3_feats)
        locs.append(self.loc[0](norm_feats))
        confs.append(self.conf[0](norm_feats))

        # conv7
        x = self.conv6(conv4_3_feats)
        x = self.conv7(x)
        locs.append(self.loc[1](x))
        confs.append(self.conf[1](x))

        # extras
        feature_map_idx = 2
        for i in range(0, len(self.extras), 2):
            x = self.extras[i](x)
            x = self.extras[i + 1](x)
            locs.append(self.loc[feature_map_idx](x))
            confs.append(self.conf[feature_map_idx](x))
            feature_map_idx += 1

        # Flatten and concatenate predictions
        locs = tf.concat([tf.reshape(l, (tf.shape(l)[0], -1, 4)) for l in locs], axis=1)
        confs = tf.concat([tf.reshape(c, (tf.shape(c)[0], -1, self.num_classes)) for c in confs], axis=1)

        return locs, confs


# Example usage
if __name__ == "__main__":
    num_classes = 21  # 20 classes + background
    ssd = SSD(num_classes)
    dummy_input = tf.random.normal([1, 300, 300, 3])
    locs, confs = ssd(dummy_input)
    print("Localization predictions shape:", locs.shape)
    print("Confidence predictions shape:", confs.shape)
