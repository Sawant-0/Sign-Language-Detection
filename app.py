"""
Custom Sign Language Detection System - Main Streamlit Application.
Provides Home Dashboard, Add New Signs Studio, and Live Sign Detection Interface.
"""

import time
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.landmarks import DualHandLandmarkExtractor, FEATURE_VECTOR_DIM
from src.dataset import CustomSignDataset, normalize_label, DEFAULT_DATASET_FILE
from src.model import CustomSignClassifier, DEFAULT_MODEL_PATH
from src.camera import CameraManager
from src.recognizer import RealTimeRecognizer

# Page Configuration
st.set_page_config(
    page_title="SignSense - Custom Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    /* Stat Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 36, 48, 0.8), rgba(20, 24, 33, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #00d26a;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.88rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Navigation Card */
    .nav-card {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.95), rgba(45, 55, 72, 0.85));
        border: 1px solid rgba(66, 153, 225, 0.3);
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .nav-card:hover {
        border-color: #3182ce;
        transform: translateY(-2px);
    }
    .status-badge-ready {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid #48bb78;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-warning {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(237, 137, 54, 0.2);
        color: #ed8936;
        border: 1px solid #ed8936;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-danger {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(229, 62, 62, 0.2);
        color: #e53e3e;
        border: 1px solid #e53e3e;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_dataset() -> CustomSignDataset:
    return CustomSignDataset(DEFAULT_DATASET_FILE)


@st.cache_resource
def get_extractor() -> DualHandLandmarkExtractor:
    return DualHandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.48,
        min_tracking_confidence=0.45,
    )


def load_current_model() -> Optional[CustomSignClassifier]:
    return CustomSignClassifier.load(DEFAULT_MODEL_PATH)


# Initialize Session State
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "recognition_history" not in st.session_state:
    st.session_state.recognition_history = []


# Sidebar Navigation
with st.sidebar:
    st.title("🤟 SignSense ML")
    st.caption("Custom Real-Time Sign Language Recognition")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        options=["🏠 Dashboard", "➕ Add New Signs & Training", "🔴 Live Sign Detection"],
        index=0 if st.session_state.active_page == "Dashboard"
        else 1 if st.session_state.active_page == "Add Signs"
        else 2,
    )

    if selected_page == "🏠 Dashboard":
        st.session_state.active_page = "Dashboard"
    elif selected_page == "➕ Add New Signs & Training":
        st.session_state.active_page = "Add Signs"
    else:
        st.session_state.active_page = "Live Detection"

    st.divider()
    st.caption("System Status")
    dataset = get_dataset()
    current_model = load_current_model()

    if current_model is None:
        st.markdown('<span class="status-badge-danger">Model: Not Trained</span>', unsafe_allow_html=True)
    elif current_model.is_stale(dataset):
        st.markdown('<span class="status-badge-warning">Model: Needs Retraining</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-ready">Model: Ready</span>', unsafe_allow_html=True)

    st.markdown(f"**Custom Classes:** {len(dataset.get_classes())}")
    st.markdown(f"**Total Samples:** {len(dataset.samples)}")


# -------------------------------------------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# -------------------------------------------------------------------------------------------------
if st.session_state.active_page == "Dashboard":
    st.title("SignSense: Custom Sign Language Detection")
    st.markdown(
        "Learn and recognize user-defined sign language gestures in real time using computer vision and machine learning."
    )

    dataset = get_dataset()
    current_model = load_current_model()
    stats = dataset.get_statistics()

    st.write("")

    # Top Metric Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_classes']}</div>
                <div class="metric-label">Custom Sign Classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_samples']}</div>
                <div class="metric-label">Total Saved Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        if current_model is None:
            status_text = "NOT TRAINED"
            status_color = "#e53e3e"
        elif current_model.is_stale(dataset):
            status_text = "NEEDS RETRAIN"
            status_color = "#ed8936"
        else:
            status_text = "READY"
            status_color = "#48bb78"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {status_color}; font-size: 1.6rem;">{status_text}</div>
                <div class="metric-label">Model Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        if current_model and current_model.trained_at:
            time_str = time.strftime("%H:%M:%S (%b %d)", time.localtime(current_model.trained_at))
        else:
            time_str = "None"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.35rem; color: #63b3ed;">{time_str}</div>
                <div class="metric-label">Last Training</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Prominent Navigation Cards
    nav_col1, nav_col2 = st.columns(2)

    with nav_col1:
        st.markdown(
            """
            <div class="nav-card">
                <h3>➕ Add New Signs & Dataset Studio</h3>
                <p style="color: #cbd5e0;">
                    Define custom sign names, record webcam landmark samples for both hands,
                    manage saved classes, and train or retrain your machine learning classifier.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Dataset & Training Studio →", use_container_width=True, type="primary"):
            st.session_state.active_page = "Add Signs"
            st.rerun()

    with nav_col2:
        model_ready = current_model is not None and not current_model.is_stale(dataset)
        badge = "🟢 Model Ready" if model_ready else "⚠️ Model Needs Training"
        st.markdown(
            f"""
            <div class="nav-card">
                <h3>🔴 Live Sign Detection</h3>
                <p style="color: #cbd5e0;">
                    Connect your webcam for real-time sign recognition using the latest trained model,
                    dual-hand landmark tracking, confidence thresholding, and prediction history.
                </p>
                <p style="margin-top: 0.5rem; font-weight: 600; color: {'#48bb78' if model_ready else '#ed8936'};">
                    {badge}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Live Detection Interface →", use_container_width=True):
            st.session_state.active_page = "Live Detection"
            st.rerun()

    st.write("")
    st.divider()

    # Dataset Distribution Overview
    st.subheader("Dataset Class Distribution")
    sample_counts = stats["sample_counts"]

    if not sample_counts:
        st.info("The dataset is currently empty. Click **'Add New Signs'** to begin creating custom signs.")
    else:
        df_counts = pd.DataFrame([{"Sign": k, "Samples": v} for k, v in sample_counts.items()])
        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            st.bar_chart(df_counts.set_index("Sign"))
        with table_col:
            st.dataframe(df_counts, hide_index=True, use_container_width=True)

        if len(sample_counts) < 2:
            st.warning("⚠️ You need at least **2 distinct sign classes** with at least 3 samples each to train the model.")
        elif current_model and current_model.is_stale(dataset):
            st.warning("⚠️ Changes have been made to the dataset since the last training run. Please retrain your model.")


# -------------------------------------------------------------------------------------------------
# PAGE 2: ADD NEW SIGNS & TRAINING INTERFACE
# -------------------------------------------------------------------------------------------------
elif st.session_state.active_page == "Add Signs":
    st.title("➕ Dataset Collection & Training Studio")
    st.markdown(
        "Create custom signs, capture hand landmark samples via webcam, manage existing labels, and train your model."
    )

    dataset = get_dataset()
    extractor = get_extractor()

    tab_record, tab_manage, tab_train = st.tabs([
        "📸 Record New Samples",
        "📂 Manage Existing Classes",
        "🧠 Train / Retrain Model",
    ])

    # ---------------- TAB 1: RECORD SAMPLES ----------------
    with tab_record:
        st.subheader("1. Select or Create Sign Label")
        existing_classes = dataset.get_classes()

        create_col1, create_col2 = st.columns([2, 1])
        with create_col1:
            target_sign = st.text_input(
                "Sign Name (Word, Phrase, or Gesture)",
                placeholder="e.g. HELLO, THANK YOU, WATER, HELP, YES, NO",
                help="Enter any sign name. Letters will be normalized automatically.",
            )
        with create_col2:
            gesture_mode = st.selectbox(
                "Gesture Type",
                options=["static", "dynamic"],
                format_func=lambda x: "Static Pose (Single-frame)" if x == "static" else "Dynamic Movement (Sequence)",
            )

        norm_target = normalize_label(target_sign)
        if norm_target and norm_target in existing_classes:
            st.info(f"ℹ️ Class **'{norm_target}'** already exists. New recordings will add more training examples to it.")

        st.divider()
        st.subheader("2. Webcam Recording Controls")

        cam_col, ctrl_col = st.columns([3, 2])

        with ctrl_col:
            st.markdown("#### Recording Settings")
            if gesture_mode == "static":
                num_samples_to_capture = st.slider(
                    "Samples to capture in burst", min_value=5, max_value=50, value=20, step=5
                )
                st.caption("Position your hand(s) steadily. The camera will capture the requested number of valid frames.")
            else:
                seq_frames = st.slider(
                    "Sequence length (frames)", min_value=15, max_value=45, value=25, step=5
                )
                st.caption("Perform the movement smoothly while the camera records the temporal sequence.")

            hand_requirement = st.radio(
                "Hands Required for this Sign",
                options=["Both Hands (Two-Handed)", "Single Hand (One-Handed)", "Any Hand (Flexible)"],
                index=0 if (norm_target in ["WATER", "PLEASE", "HOME", "HELLO HOME"] or " " in norm_target) else 2,
                help="Ensures every captured sample strictly has the required hand(s). Eliminates noisy frames and boosts accuracy.",
            )

            st.write("")
            start_cam_btn = st.button("🎥 Start Webcam Feed", type="primary", use_container_width=True)
            stop_cam_btn = st.button("⏹️ Stop Camera", use_container_width=True)

            status_placeholder = st.empty()
            progress_bar = st.empty()

        with cam_col:
            frame_placeholder = st.empty()

        if start_cam_btn:
            if not norm_target:
                status_placeholder.error("Please enter a valid sign name before recording.")
            else:
                camera = CameraManager(device_index=0)
                cam_ok, cam_msg = camera.start()

                if not cam_ok:
                    status_placeholder.error(cam_msg)
                else:
                    status_placeholder.info(f"Camera active. Recording samples for '{norm_target}'...")
                    captured_count = 0
                    dynamic_sequence_buffer = []

                    # 3-second countdown before recording
                    for countdown in range(3, 0, -1):
                        ret, raw_frame = camera.read_frame()
                        if ret and raw_frame is not None:
                            h, w, _ = raw_frame.shape
                            cv2.putText(
                                raw_frame,
                                f"Get Ready: {countdown}",
                                (w // 4, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 220, 255),
                                3,
                            )
                            frame_placeholder.image(raw_frame, channels="BGR", use_column_width=True)
                        time.sleep(0.8)

                    # Capture Loop
                    target_count = num_samples_to_capture if gesture_mode == "static" else seq_frames
                    start_time = time.time()

                    while captured_count < target_count and (time.time() - start_time) < 30.0:
                        ret, raw_frame = camera.read_frame()
                        if not ret or raw_frame is None:
                            continue

                        # Extract dual-hand landmarks
                        features, left_det, right_det, annotated_frame, meta = extractor.extract_features(
                            raw_frame, draw=True
                        )

                        both_det = meta["both_hands_detected"]
                        any_det = left_det or right_det

                        if hand_requirement.startswith("Both Hands"):
                            is_valid_frame = both_det
                            warning_banner = "Waiting for BOTH hands..." if not both_det else None
                        elif hand_requirement.startswith("Single Hand"):
                            is_valid_frame = (left_det and not right_det) or (right_det and not left_det)
                            warning_banner = "Waiting for a SINGLE hand..." if not is_valid_frame else None
                        else:
                            is_valid_frame = any_det
                            warning_banner = "Waiting for hands..." if not any_det else None

                        if gesture_mode == "static":
                            if is_valid_frame:
                                save_ok, _ = dataset.add_static_sample(norm_target, features)
                                if save_ok:
                                    captured_count += 1
                                    progress_bar.progress(captured_count / target_count)
                        else:
                            # Dynamic sequence: buffer frame only when valid hands present
                            if is_valid_frame:
                                dynamic_sequence_buffer.append(features)
                                captured_count += 1
                                progress_bar.progress(captured_count / target_count)

                        # Overlay recording status on preview
                        h, w, _ = annotated_frame.shape
                        cv2.putText(
                            annotated_frame,
                            f"RECORDING: {captured_count}/{target_count}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2,
                        )
                        if warning_banner:
                            cv2.putText(
                                annotated_frame,
                                warning_banner,
                                (20, h - 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 165, 255),
                                2,
                            )

                        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                        time.sleep(0.04)

                    camera.stop()

                    if gesture_mode == "dynamic" and len(dynamic_sequence_buffer) >= 10:
                        save_ok, msg = dataset.add_dynamic_sample(norm_target, dynamic_sequence_buffer)
                        if save_ok:
                            status_placeholder.success(f"✅ Successfully recorded dynamic sequence for '{norm_target}'!")
                        else:
                            status_placeholder.error(f"❌ Recording failed: {msg}")
                    elif gesture_mode == "static" and captured_count > 0:
                        status_placeholder.success(f"✅ Successfully saved {captured_count} samples for '{norm_target}'!")
                    else:
                        status_placeholder.warning("⚠️ No valid hands detected during recording. Please try again.")

                    st.rerun()

    # ---------------- TAB 2: MANAGE CLASSES ----------------
    with tab_manage:
        st.subheader("Existing Sign Classes in Dataset")
        existing_classes = dataset.get_classes()
        counts = dataset.get_sample_counts()

        if not existing_classes:
            st.info("No custom sign classes found in dataset.")
        else:
            table_data = []
            breakdown = dataset.get_hand_breakdown()
            inconsistent_classes = []

            for cls in existing_classes:
                cls_meta = dataset.classes.get(cls, {})
                cls_bd = breakdown.get(cls, {})
                both_c = cls_bd.get("both", 0)
                left_c = cls_bd.get("left_only", 0)
                right_c = cls_bd.get("right_only", 0)
                dyn_c = cls_bd.get("dynamic", 0)
                tot_c = counts.get(cls, 0)

                # Check consistency
                if both_c > 0 and (left_c > 0 or right_c > 0):
                    inconsistent_classes.append(cls)
                    status_str = "⚠️ Mixed Hands"
                elif both_c > 0:
                    status_str = "🟢 Both Hands"
                elif left_c > 0 or right_c > 0:
                    status_str = "🟢 Single Hand"
                elif dyn_c > 0:
                    status_str = "🟢 Dynamic Movement"
                else:
                    status_str = "⚪ Empty"

                table_data.append({
                    "Sign Name": cls,
                    "Total Samples": tot_c,
                    "Both Hands": both_c,
                    "Left Only": left_c,
                    "Right Only": right_c,
                    "Dynamic": dyn_c,
                    "Quality Status": status_str,
                    "Type": cls_meta.get("mode", "static").capitalize(),
                })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            if inconsistent_classes:
                st.warning(
                    f"⚠️ **Inconsistent hand counts detected in: {', '.join(inconsistent_classes)}**\n\n"
                    "These signs have a mix of two-handed and single-handed samples (usually caused when MediaPipe dropped a hand during recording). "
                    "Cleaning accidental single-hand frames ensures clean training features and significantly improves accuracy."
                )
                if st.button("🧹 Auto-Clean & Align Inconsistent Samples", type="primary"):
                    removed, clean_msg = dataset.clean_inconsistent_samples()
                    st.success(f"🎉 {clean_msg} Remember to retrain your model in the 'Train' tab.")
                    st.rerun()

            st.divider()
            m_col1, m_col2 = st.columns(2)

            with m_col1:
                st.markdown("#### Rename a Class")
                rename_target = st.selectbox("Select sign to rename", options=existing_classes, key="rename_select")
                new_name = st.text_input("New Name", key="rename_input")
                if st.button("Apply Rename", key="rename_btn"):
                    ok, msg = dataset.rename_class(rename_target, new_name)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            with m_col2:
                st.markdown("#### Delete a Class")
                delete_target = st.selectbox("Select sign to delete", options=existing_classes, key="delete_select")
                confirm_del = st.checkbox(f"Confirm deletion of '{delete_target}' and all its samples", key="confirm_del")
                if st.button("Delete Sign Class", type="primary", disabled=not confirm_del, key="delete_btn"):
                    ok, msg = dataset.delete_class(delete_target)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            st.write("")
            with st.expander("Danger Zone: Clear Dataset"):
                st.warning("This will permanently delete all custom signs and training samples.")
                if st.button("Clear Entire Dataset", type="secondary"):
                    dataset.clear_all()
                    st.success("Dataset cleared.")
                    st.rerun()

    # ---------------- TAB 3: TRAIN MODEL ----------------
    with tab_train:
        st.subheader("Model Training & Evaluation Pipeline")
        existing_classes = dataset.get_classes()
        sample_counts = dataset.get_sample_counts()
        total_samples = len(dataset.samples)

        st.markdown(
            f"""
            - **Custom Classes:** {len(existing_classes)}
            - **Total Valid Samples:** {total_samples}
            - **Minimum Requirement:** At least 2 classes, each with 3 or more samples.
            """
        )

        active_classes = [c for c, cnt in sample_counts.items() if cnt >= 1]
        can_train = len(active_classes) >= 2

        if not can_train:
            if len(existing_classes) < 2:
                st.warning("⚠️ At least 2 distinct sign classes are required before training.")
            else:
                st.warning("⚠️ At least 2 sign classes need recorded samples before training.")

        st.write("")
        train_btn = st.button("🚀 Train / Retrain Model", type="primary", disabled=not can_train)

        if train_btn:
            with st.spinner("Training machine learning classifier on custom landmark dataset..."):
                clf = CustomSignClassifier()
                success, msg, metrics = clf.train(dataset)

                if success:
                    st.success(f"🎉 {msg}")
                    st.balloons()

                    # Display genuine evaluation metrics
                    st.subheader("Evaluation Results")
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)

                    val_acc = metrics.get("validation_accuracy", metrics.get("training_accuracy", "N/A"))
                    val_prec = metrics.get("validation_precision", "N/A")
                    val_rec = metrics.get("validation_recall", "N/A")
                    val_f1 = metrics.get("validation_f1", "N/A")

                    m_c1.metric("Validation Accuracy", f"{val_acc}%" if val_acc != "N/A" else "N/A")
                    m_c2.metric("Precision", f"{val_prec}%" if val_prec != "N/A" else "N/A")
                    m_c3.metric("Recall", f"{val_rec}%" if val_rec != "N/A" else "N/A")
                    m_c4.metric("F1-Score", f"{val_f1}%" if val_f1 != "N/A" else "N/A")

                    # Confusion Matrix Plot
                    cm = metrics.get("confusion_matrix")
                    if cm and len(cm) > 1:
                        st.write("#### Confusion Matrix")
                        classes = metrics.get("classes", [])
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Greens",
                            xticklabels=classes,
                            yticklabels=classes,
                            ax=ax,
                        )
                        ax.set_xlabel("Predicted Sign")
                        ax.set_ylabel("True Sign")
                        st.pyplot(fig)
                else:
                    st.error(f"❌ Training Failed: {msg}")


# -------------------------------------------------------------------------------------------------
# PAGE 3: LIVE SIGN DETECTION INTERFACE
# -------------------------------------------------------------------------------------------------
elif st.session_state.active_page == "Live Detection":
    st.title("🔴 Live Sign Detection")
    st.markdown("Real-time sign recognition using your latest trained custom model and webcam feed.")

    dataset = get_dataset()
    current_model = load_current_model()
    extractor = get_extractor()

    # Pre-flight check
    if current_model is None:
        sample_counts = dataset.get_sample_counts()
        active_classes = [c for c, cnt in sample_counts.items() if cnt >= 1]
        if len(active_classes) >= 2:
            st.warning(
                f"⚠️ A dataset with {len(active_classes)} signs (**{', '.join(active_classes)}**) is ready, "
                f"but the classifier has not been trained yet."
            )
            if st.button("⚡ Train Model Now & Start Recognition", type="primary"):
                with st.spinner("Training machine learning model on your recorded samples..."):
                    clf = CustomSignClassifier()
                    ok, msg, _ = clf.train(dataset)
                    if ok:
                        st.success(f"🎉 {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ Training failed: {msg}")
        else:
            st.error(
                "❌ No trained model found! Please go to **'Add New Signs & Training'**, "
                "record samples for at least 2 signs, and train your model first."
            )
            if st.button("Go to Dataset & Training Studio →"):
                st.session_state.active_page = "Add Signs"
                st.rerun()
    elif current_model.is_stale(dataset):
        st.warning(
            "⚠️ The dataset has been modified since the model was last trained. "
            "Live detection will use the existing model, but retraining is recommended."
        )

    if current_model is not None:
        # Recognition Configuration Controls
        st.markdown("#### Recognition Controls")
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.5, 1.5, 1, 1])

        with ctrl_c1:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.40,
                max_value=0.95,
                value=0.60,
                step=0.05,
                help="Predictions with confidence below this threshold are marked as 'Low Confidence / Unknown'.",
            )
        with ctrl_c2:
            det_sensitivity = st.slider(
                "Hand Detection Sensitivity",
                min_value=0.30,
                max_value=0.75,
                value=0.45,
                step=0.05,
                help="Lower value detects both hands more aggressively in low lighting or tricky angles.",
            )
        with ctrl_c3:
            cam_start_btn = st.button("▶️ Start Camera Feed", type="primary", use_container_width=True)
        with ctrl_c4:
            cam_stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

        st.divider()

        # Two-column layout: Video Feed on left, Real-time Dashboard on right
        video_col, hud_col = st.columns([3, 2])

        with video_col:
            video_frame_placeholder = st.empty()

        with hud_col:
            st.markdown("#### Current Prediction")
            pred_badge_placeholder = st.empty()
            confidence_bar_placeholder = st.empty()

            st.write("")
            st.markdown("#### Class Probabilities")
            prob_chart_placeholder = st.empty()

            st.write("")
            st.markdown("#### Recognition History")
            history_placeholder = st.empty()
            if st.button("Clear History", use_container_width=True):
                st.session_state.recognition_history = []
                st.rerun()

        # Live Webcam Loop
        if cam_start_btn:
            extractor.set_sensitivity(min_detection_confidence=det_sensitivity, min_tracking_confidence=0.40)
            camera = CameraManager(device_index=0)
            cam_ok, cam_msg = camera.start()

            if not cam_ok:
                st.error(cam_msg)
            else:
                recognizer = RealTimeRecognizer(
                    classifier=current_model,
                    extractor=extractor,
                    confidence_threshold=conf_threshold,
                )

                st.session_state.camera_active = True
                stop_requested = False

                while st.session_state.camera_active and not stop_requested:
                    ret, raw_frame = camera.read_frame()
                    if not ret or raw_frame is None:
                        continue

                    # Process frame through recognition orchestrator
                    res, annotated_frame = recognizer.process_frame(raw_frame, draw=True)

                    # Update video placeholder
                    video_frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

                    # Update prediction badge
                    stable_sign = res.get("stable_prediction")
                    conf = res.get("confidence", 0.0)
                    status = res.get("status", "")

                    if stable_sign:
                        pred_badge_placeholder.markdown(
                            f"""
                            <div style="background-color: rgba(0, 210, 106, 0.15); border: 2px solid #00d26a; border-radius: 10px; padding: 12px; text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: 800; color: #00d26a;">{stable_sign}</div>
                                <div style="color: #a0aec0; font-size: 0.9rem;">Confidence: {int(conf * 100)}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        confidence_bar_placeholder.progress(min(1.0, conf))
                    elif status == "LOW_CONFIDENCE":
                        pred_badge_placeholder.markdown(
                            """
                            <div style="background-color: rgba(237, 137, 54, 0.15); border: 2px solid #ed8936; border-radius: 10px; padding: 12px; text-align: center;">
                                <div style="font-size: 1.4rem; font-weight: 700; color: #ed8936;">Low Confidence</div>
                                <div style="color: #a0aec0; font-size: 0.9rem;">Below threshold (Unknown Sign)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        confidence_bar_placeholder.progress(0.0)
                    elif status == "NO_HANDS_DETECTED":
                        pred_badge_placeholder.markdown(
                            """
                            <div style="background-color: rgba(160, 174, 192, 0.1); border: 1px solid #718096; border-radius: 10px; padding: 12px; text-align: center;">
                                <div style="font-size: 1.2rem; font-weight: 600; color: #cbd5e0;">No Hands Detected</div>
                                <div style="color: #a0aec0; font-size: 0.85rem;">Place hands in camera view</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        confidence_bar_placeholder.progress(0.0)

                    # Update probabilities chart
                    probs = res.get("probabilities", {})
                    if probs:
                        df_probs = pd.DataFrame([{"Sign": k, "Probability": v} for k, v in probs.items()])
                        prob_chart_placeholder.bar_chart(df_probs.set_index("Sign"), height=160)

                    # Update history
                    if recognizer.history:
                        st.session_state.recognition_history = recognizer.history
                        df_hist = pd.DataFrame(recognizer.history[-6:][::-1])
                        history_placeholder.dataframe(
                            df_hist[["time_str", "sign", "confidence"]].rename(
                                columns={"time_str": "Time", "sign": "Sign", "confidence": "Confidence"}
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )

                    time.sleep(0.03)

                camera.stop()
