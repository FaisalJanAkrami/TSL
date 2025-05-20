import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, LayerNormalization
from collections import deque, Counter
import time

# ✅ SETTINGS
NUM_FRAMES = 32
RAW_DIM = 33 * 3 + 21 * 3 * 2  # pose + lh + rh
ANGLE_DIM = 10
INPUT_DIM = RAW_DIM * 2 + ANGLE_DIM
SELECTED_CLASSES = ['beklemek', 'ben', 'ev', 'yemek', 'yapmak']
STABILITY_WINDOW = 5  # number of past predictions to average

# ✅ TEMPORAL ATTENTION
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, bias_init=None, **kwargs):
        super().__init__(**kwargs)
        self.bias_init = bias_init

    def build(self, input_shape):
        _, timesteps, _ = input_shape
        self.attn_weights = self.add_weight(
            shape=(timesteps,),
            initializer="zeros",
            trainable=True,
            name="attn_weights"
        )
        if self.bias_init is not None:
            self.attn_weights.assign_add(tf.reduce_mean(self.bias_init, axis=-1))

    def call(self, inputs):
        attn_scores = tf.nn.softmax(self.attn_weights)
        return tf.reduce_sum(inputs * tf.expand_dims(attn_scores, -1), axis=1)

# ✅ MODEL
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = TemporalAttention()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

model = build_model((NUM_FRAMES, INPUT_DIM), len(SELECTED_CLASSES))
model.load_weights("best_model.h5")

# ✅ Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    def safe_landmarks(landmarks, count):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        else:
            return np.zeros(count * 3)
    pose = safe_landmarks(results.pose_landmarks.landmark if results.pose_landmarks else None, 33)
    lh = safe_landmarks(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, 21)
    rh = safe_landmarks(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, 21)
    return np.concatenate([pose, lh, rh])  # (225,)

def compute_velocity(sequence):
    return np.diff(sequence, axis=0, prepend=sequence[:1])

def compute_angles(sequence):
    angles_seq = []
    for frame in sequence:
        angles = []
        pose = frame[:99].reshape(33, 3)
        lh = frame[99:99+63].reshape(21, 3)
        rh = frame[162:162+63].reshape(21, 3)
        for hand in [lh, rh]:
            for a, b, c in [(0, 5, 8), (0, 9, 12), (0, 13, 16), (0, 17, 20)]:
                vec1 = hand[b] - hand[a]
                vec2 = hand[c] - hand[b]
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                angles.append(cos_theta)
        for a, b, c in [(11, 13, 15), (12, 14, 16)]:
            vec1 = pose[b] - pose[a]
            vec2 = pose[c] - pose[b]
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            angles.append(cos_theta)
        angles_seq.append(angles)
    return np.array(angles_seq)

# ✅ Live Prediction Loop
def predict_from_webcam():
    print("📷 Starting webcam stream...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    sequence = deque(maxlen=NUM_FRAMES)
    recent_preds = deque(maxlen=STABILITY_WINDOW)
    final_prediction = None

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == NUM_FRAMES:
                raw_seq = np.array(sequence)
                vel_seq = compute_velocity(raw_seq)
                ang_seq = compute_angles(raw_seq)
                full_input = np.concatenate([raw_seq, vel_seq, ang_seq], axis=-1)[None, ...]

                prediction = model.predict(full_input, verbose=0)
                sign_id = np.argmax(prediction)
                confidence = np.max(prediction)
                sign_name = SELECTED_CLASSES[sign_id]

                if confidence > 0.98:
                    recent_preds.append(sign_id)
                    most_common = Counter(recent_preds).most_common(1)[0]
                    if most_common[1] >= STABILITY_WINDOW:
                        final_prediction = sign_name
                else:
                    recent_preds.clear()

            label = final_prediction if final_prediction else "Detecting..."
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow("TSL Live Recognition", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Run
predict_from_webcam()
