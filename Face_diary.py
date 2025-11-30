import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


DATA_DIR = "face_diary_data"
IMG_DIR = os.path.join(DATA_DIR, "images")
LOG_PATH = os.path.join(DATA_DIR, "log.csv")


def ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)


def compute_face_features(landmarks):
    """
    landmarks: list of mediapipe landmarks (x, y, z all in [0,1] normalized coords)
    returns: dict of simple numeric features
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]

    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    aspect = float(width / height) if height > 0 else 0.0
    mean_depth = float(zs.mean())

    # how spread out the landmarks are around the center (rough shape/pose indicator)
    center_xy = pts[:, :2].mean(axis=0)
    spread = float(np.linalg.norm(pts[:, :2] - center_xy, axis=1).mean())

    return {
        "face_width": width,
        "face_height": height,
        "aspect_ratio": aspect,
        "mean_depth": mean_depth,
        "avg_spread": spread,
    }


def append_log_row(timestamp_str, image_name, features):
    row = {"timestamp": timestamp_str, "image": image_name}
    row.update(features)

    df_row = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH)
    df_row.to_csv(LOG_PATH, mode="a", header=header, index=False)


def capture_session():
    ensure_dirs()

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Face diary capture started.")
    print("Press 'c' to capture an entry, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh.process(rgb)

        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    lm,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    draw_spec,
                    draw_spec,
                )

        cv2.imshow("Face Diary (press c to capture, q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            if not res.multi_face_landmarks:
                print("no face detected, not logging this frame.")
                continue

            landmarks = res.multi_face_landmarks[0].landmark
            features = compute_face_features(landmarks)

            now = datetime.now()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
            filename = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
            img_path = os.path.join(IMG_DIR, filename)
            cv2.imwrite(img_path, frame)

            append_log_row(ts_str, filename, features)

            print(f"logged entry at {ts_str} -> {filename}")
            print(f"features: {features}")

    cap.release()
    cv2.destroyAllWindows()
    print("capture session ended.")


def analyze_diary():
    if not os.path.exists(LOG_PATH):
        print("no log found yet. run capture mode first.")
        return

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        print("log is empty.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feature_cols = ["face_width", "face_height", "aspect_ratio", "mean_depth", "avg_spread"]
    print("\nbasic stats:")
    print(df[feature_cols].describe())

    # simple plot: face_width over time
    plt.figure()
    plt.plot(df["timestamp"], df["face_width"], marker="o")
    plt.xlabel("time")
    plt.ylabel("face_width (normalized)")
    plt.title("face width over time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # tiny clustering step for fun (unsupervised "ML" on the features)
    if len(df) >= 3:
        k = 3 if len(df) >= 6 else 2
        X = df[feature_cols].values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X)

        print(f"\ncluster assignments (k={k}):")
        print(df[["timestamp", "image", "cluster"]])

        # optional: save updated log with clusters
        df.to_csv(LOG_PATH, index=False)
    else:
        print("\nnot enough entries yet for clustering (need at least 3).")


def main():
    mode = input("mode [capture / analyze] (default=capture): ").strip().lower()
    if mode == "" or mode == "capture":
        capture_session()
    elif mode == "analyze":
        analyze_diary()
    else:
        print("unknown mode, use 'capture' or 'analyze'.")


if __name__ == "__main__":
    main()
