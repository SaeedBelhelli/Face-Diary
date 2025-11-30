import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.cluster import KMeans

mp_face_mesh = mp.solutions.face_mesh

FEATURE_COLS = ["face_width", "face_height", "aspect_ratio", "mean_depth", "avg_spread"]


def compute_face_features(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]

    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    aspect = float(width / height) if height > 0 else 0.0
    mean_depth = float(zs.mean())

    center_xy = pts[:, :2].mean(axis=0)
    spread = float(np.linalg.norm(pts[:, :2] - center_xy, axis=1).mean())

    return {
        "face_width": width,
        "face_height": height,
        "aspect_ratio": aspect,
        "mean_depth": mean_depth,
        "avg_spread": spread,
    }


def process_image(image_bgr):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None, None

    landmarks = res.multi_face_landmarks[0].landmark

    # draw landmarks for visualization
    img_vis = image_bgr.copy()
    h, w, _ = img_vis.shape
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(img_vis, (x, y), 1, (0, 255, 0), -1)

    features = compute_face_features(landmarks)
    img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
    return img_vis_rgb, features


def init_state():
    if "log" not in st.session_state:
        st.session_state["log"] = pd.DataFrame(columns=["timestamp", "source"] + FEATURE_COLS)


def add_entry(source, features):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": ts, "source": source}
    row.update(features)
    st.session_state["log"] = pd.concat(
        [st.session_state["log"], pd.DataFrame([row])],
        ignore_index=True,
    )


def main():
    st.set_page_config(page_title="Face Diary", layout="wide")
    init_state()

    st.title("Face Diary – live demo")

    tab_capture, tab_history = st.tabs(["Capture entry", "History / ML"])

    # ---- Capture tab ----
    with tab_capture:
        st.write("Capture a face using your camera, or use a sample image.")

        mode = st.radio("Input source", ["Camera", "Sample image"], horizontal=True)

        img_bgr = None

        if mode == "Camera":
            cam_image = st.camera_input("Take a photo")
            if cam_image is not None:
                file_bytes = np.asarray(bytearray(cam_image.getvalue()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            st.image("sample_face.jpg", caption="Sample image", use_container_width=False)
            if st.button("Use sample image"):
                img_bgr = cv2.imread("sample_face.jpg")

        if img_bgr is not None:
            st.write("Processing face…")
            vis_rgb, features = process_image(img_bgr)

            if features is None:
                st.warning("No face detected. Try again with a clearer face.")
            else:
                add_entry(mode.lower(), features)
                st.image(vis_rgb, caption="Detected face landmarks", use_container_width=False)
                st.success("Entry added to diary.")
                st.json(features)

    # ---- History / ML tab ----
    with tab_history:
        df = st.session_state["log"]
        if df.empty:
            st.info("No entries yet. Capture one in the other tab.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.subheader("Logged entries")
        st.dataframe(df)

        st.subheader("Feature over time")
        feature = st.selectbox("Choose feature", FEATURE_COLS, index=0)
        st.line_chart(df.set_index("timestamp")[feature])

        st.subheader("KMeans clustering on feature vectors")
        if len(df) < 3:
            st.info("Need at least 3 entries for clustering.")
            return

        k_default = 3 if len(df) >= 6 else 2
        k = st.slider("Number of clusters (k)", 2, min(6, len(df)), k_default)

        X = df[FEATURE_COLS].values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X)

        st.write("Cluster counts:")
        st.bar_chart(df["cluster"].value_counts().sort_index())

        st.subheader("Scatter (two features, colored by cluster)")
        x_feat = st.selectbox("X axis", FEATURE_COLS, index=0)
        y_feat = st.selectbox("Y axis", FEATURE_COLS, index=1)
        st.scatter_chart(df, x=x_feat, y=y_feat, color="cluster")


if __name__ == "__main__":
    main()
