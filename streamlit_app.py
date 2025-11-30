import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


mp_face_mesh = mp.solutions.face_mesh

FEATURE_COLS = ["face_width", "face_height", "aspect_ratio", "mean_depth", "avg_spread"]
MOOD_OPTIONS = ["-- no label --", "tired", "neutral", "happy", "stressed", "other"]


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
        st.session_state["log"] = pd.DataFrame(
            columns=["timestamp", "source", "mood", "note"] + FEATURE_COLS
        )


def add_entry(source, mood, note, features):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": ts, "source": source, "mood": mood, "note": note}
    row.update(features)
    st.session_state["log"] = pd.concat(
        [st.session_state["log"], pd.DataFrame([row])],
        ignore_index=True,
    )


def main():
    st.set_page_config(page_title="Face Diary", layout="wide")
    init_state()

    st.title("Face Diary – Face Features + Mood Logging")

    tab_capture, tab_diary, tab_ml = st.tabs(["Capture entry", "Diary & features", "ML models"])

    # ------------------- Capture tab -------------------
    with tab_capture:
        st.write("Capture entries with your face + mood, or use a sample image.")

        col_left, col_right = st.columns(2)

        with col_left:
            mode = st.radio("Input source", ["Camera", "Sample image"], horizontal=True)

            mood = st.selectbox("How do you feel right now?", MOOD_OPTIONS, index=0)
            note = st.text_input("Short note (optional)", "")

            img_bgr = None
            source_label = None

            if mode == "Camera":
                st.write("Set mood/note first, then take a photo.")
                cam_image = st.camera_input("Take a photo")
                if cam_image is not None:
                    file_bytes = np.asarray(bytearray(cam_image.getvalue()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    source_label = "camera"
            else:
                st.image("sample_face.jpg", caption="Sample image", use_container_width=False)
                if st.button("Use sample image as new entry"):
                    img_bgr = cv2.imread("sample_face.jpg")
                    source_label = "sample"

            if img_bgr is not None:
                st.write("Processing face…")
                vis_rgb, features = process_image(img_bgr)

                if features is None:
                    st.warning("No face detected. Try again with a clearer face.")
                else:
                    if mood not in MOOD_OPTIONS:
                        mood_used = "-- no label --"
                    else:
                        mood_used = mood

                    add_entry(source_label, mood_used, note, features)

                    st.success("Entry added to diary.")
                    st.image(vis_rgb, caption="Detected face landmarks", use_container_width=False)
                    st.json(features)

        with col_right:
            df = st.session_state["log"]
            st.subheader("Recent entries (this session)")
            if df.empty:
                st.info("No entries yet. Capture one on the left.")
            else:
                st.dataframe(df.tail(5))

    # ------------------- Diary & features tab -------------------
    with tab_diary:
        df = st.session_state["log"]
        if df.empty:
            st.info("No entries yet. Use the 'Capture entry' tab first.")
        else:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.subheader("Full diary")
            st.dataframe(df)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download diary as CSV",
                data=csv_bytes,
                file_name="face_diary_log.csv",
                mime="text/csv",
            )

            st.subheader("Feature over time")
            feature = st.selectbox("Choose feature to plot", FEATURE_COLS, index=0)
            st.line_chart(df.set_index("timestamp")[feature])

    # ------------------- ML models tab -------------------
    with tab_ml:
        df = st.session_state["log"]
        if df.empty:
            st.info("No entries yet. Use the 'Capture entry' tab first.")
            return

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.subheader("1. KMeans clustering on face features")

        if len(df) < 3:
            st.info("Need at least 3 entries for clustering.")
        else:
            max_k = min(5, len(df))
            k_default = 3 if len(df) >= 6 else 2
            k = st.slider("Number of clusters (k)", 2, max_k, k_default)

            X = df[FEATURE_COLS].values
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["cluster"] = kmeans.fit_predict(X)

            st.write("Cluster counts:")
            st.bar_chart(df["cluster"].value_counts().sort_index())

            st.write("Average feature values per cluster:")
            st.dataframe(df.groupby("cluster")[FEATURE_COLS].mean().round(4))

            st.subheader("Scatter (two features, color = cluster)")
            x_feat = st.selectbox("X axis feature", FEATURE_COLS, index=0, key="x_feat")
            y_feat = st.selectbox("Y axis feature", FEATURE_COLS, index=1, key="y_feat")

            st.scatter_chart(df, x=x_feat, y=y_feat, color="cluster")

        st.markdown("---")
        st.subheader("2. Mood classifier (supervised)")

        labelled = df[df["mood"] != "-- no label --"].copy()
        if labelled.empty or labelled["mood"].nunique() < 2:
            st.info(
                "To train a classifier, log entries with different mood labels "
                "(e.g. tired, neutral, happy)."
            )
            return

        st.write(f"Labeled entries: {len(labelled)} · Classes: {labelled['mood'].unique()}")

        X = labelled[FEATURE_COLS].values
        y = labelled["mood"].values

        if len(labelled) < 6:
            st.warning(
                "Very small dataset – training and testing on such few examples "
                "is just for demonstration."
            )
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            test_size = 0.3
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        clf = RandomForestClassifier(
            n_estimators=200, random_state=42, min_samples_leaf=1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy on held-out data (rough): **{acc:.2f}**")

        importances = clf.feature_importances_
        fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        st.write("Feature importances (higher = more influence):")
        st.bar_chart(fi_df.set_index("feature"))

        st.write("Predictions vs true moods (sample):")
        preview = pd.DataFrame(
            {
                "true_mood": y_test,
                "predicted_mood": y_pred,
            }
        )
        st.dataframe(preview)


if __name__ == "__main__":
    main()
