from keras.engine import Model
from keras.layers import Flatten, Input
from keras.callbacks import ModelCheckpoint
from model import get_model
import numpy as np

X = np.load('/home/ubuntu/CourseAI/datasets/facs/X.npy')
Y = np.load('/home/ubuntu/CourseAI/datasets/facs/Y.npy')

def generator(seq_ids, seq_len, batch_size=2):
    sample_size = len(seq_ids)
    k = 0
    while True:
        frame_batch = np.zeros((batch_size, seq_len, 224, 224, 3))
        landmark_batch = np.zeros((batch_size, seq_len, 68, 2))
        labels_batch = np.zeros((batch_size, 65), dtype=np.float32)

        for b in range(batch_size):
            k += 1
            k %= sample_size
            if not k:
                np.random.shuffle(seq_ids)

            seq_id = seq_ids[k]
            seq_frames = X[seq_id][0]
            start_id = len(seq_frames) - seq_len if len(seq_frames) > seq_len else 0

            frames = X[seq_id][0][start_id:start_id + seq_len]
            landmarks = X[seq_id][1][start_id:start_id + seq_len]

            while len(frames) != seq_len:
                frames.append(frames[-1])
                landmarks.append(landmarks[-1])

            frame_batch[b, ...] = frames
            landmark_batch[b, ...] = landmarks

            labels_batch[b, ...] = Y[seq_id]

            k = min(sample_size - 1, k)

        frame_batch[:, :, :, :, 0] -= 93.5940
        frame_batch[:, :, :, :, 1] -= 104.7624
        frame_batch[:, :, :, :, 2] -= 129.1863

        yield [frame_batch, landmark_batch], labels_batch

def main():
    seq_len = 10
    batch_size = 16
    train_test_split = 0.9

    all_seq_ids = np.arange(len(X))
    np.random.shuffle(all_seq_ids)

    train_len = int(len(X) * train_test_split)
    test_len = len(X) - train_len

    train_gen = generator(all_seq_ids[:train_len], seq_len, batch_size=batch_size)
    test_gen = generator(all_seq_ids[train_len:], seq_len, batch_size=batch_size)

    model = get_model(frameCount=seq_len, nb_classes=65)
    model.summary()
    # model.load_weights('./weights/facsModel.hdf5')
    model.fit_generator(
        generator=train_gen, validation_data=test_gen,
        steps_per_epoch=int(train_len / batch_size),
        validation_steps=int(test_len / batch_size),
        epochs=5,
        verbose=1,
        callbacks=[
            ModelCheckpoint('./weights/facsModel.hdf5', verbose=1, monitor='val_loss', save_best_only=True)
        ])
    print('Learning is done.')

if __name__ == '__main__':
    main()