import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration

import mediapipe as mp

# reutiliza seus módulos (se o app estiver na mesma pasta src/)
from settings import (
    ANGLE_UP_THRESHOLD, ANGLE_DOWN_THRESHOLD,
    MIN_FRAMES_IN_STATE, SMOOTHING_WINDOW,
    USE_HIP_CHECK, HIP_MIN_DELTA,
    DEFAULT_TARGET_REPS, INACTIVITY_SECONDS,
)
from session_logger import SessionSummary, save_session

# ------------- Util -------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def calc_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

def fmt_time(ts): 
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

# ------------- State init -------------
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
    st.session_state.reps_total = 0
    st.session_state.sets_total = 0
    st.session_state.reps_current_set = 0
    st.session_state.reps_per_set = []
    st.session_state.last_movement_time = time.time()
    st.session_state.last_rep_time = None
    st.session_state.rep_times = []
    st.session_state.state = "TOP"
    st.session_state.frames_in_state = 0
    st.session_state.angle_hist = deque(maxlen=SMOOTHING_WINDOW)
    st.session_state.hip_y_hist = deque(maxlen=30)
    st.session_state.hip_move_ok = True
    st.session_state.angle_display = None

# ------------- Sidebar (UI) -------------
st.sidebar.title("⚙️ Configurações")

angle_up = st.sidebar.slider("Ângulo topo (°)", 140, 180, ANGLE_UP_THRESHOLD, 1)
angle_down = st.sidebar.slider("Ângulo fundo (°)", 30, 120, ANGLE_DOWN_THRESHOLD, 1)
min_frames = st.sidebar.slider("Frames mínimos por estado", 1, 10, MIN_FRAMES_IN_STATE, 1)
smooth_win = st.sidebar.slider("Suavização (janela média)", 1, 15, SMOOTHING_WINDOW, 1)
use_hip = st.sidebar.checkbox("Exigir movimento do quadril", value=USE_HIP_CHECK)
hip_min_delta = st.sidebar.slider("Amplitude mínima do quadril", 0.0, 0.1, HIP_MIN_DELTA, 0.005)
target_reps = st.sidebar.number_input("Reps alvo por série", 1, 200, DEFAULT_TARGET_REPS, 1)
idle_secs = st.sidebar.slider("Ociosidade p/ fechar série (s)", 2, 15, int(INACTIVITY_SECONDS), 1)

reset_btn = st.sidebar.button("🔁 Reset sessão")

if reset_btn:
    st.session_state.session_start = time.time()
    st.session_state.reps_total = 0
    st.session_state.sets_total = 0
    st.session_state.reps_current_set = 0
    st.session_state.reps_per_set = []
    st.session_state.last_movement_time = time.time()
    st.session_state.last_rep_time = None
    st.session_state.rep_times = []
    st.session_state.state = "TOP"
    st.session_state.frames_in_state = 0
    st.session_state.angle_hist = deque(maxlen=smooth_win)
    st.session_state.hip_y_hist = deque(maxlen=30)
    st.session_state.hip_move_ok = True
    st.session_state.angle_display = None

# ------------- Header -------------
st.title("Contador de Flexões — Streamlit v1")

# KPIs topo
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reps", st.session_state.reps_total)
col2.metric("Série atual", st.session_state.sets_total + 1)
col3.metric("Na série", f"{st.session_state.reps_current_set}/{target_reps}")
col4.metric("Estado", st.session_state.state)

# Barra de progresso de repetição
prog = min(1.0, max(0.0, (st.session_state.angle_display or 0.0 - angle_down) / (angle_up - angle_down) if angle_up != angle_down else 0.0))
st.progress(prog)

# ------------- WebRTC Video -------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class PushUpProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            smooth_landmarks=True,
        )
        self.prev_time = time.time()

    def _count_logic(self, angle_s, hip_move_ok):
        # usa session_state para manter contadores entre frames
        st.session_state.frames_in_state += 1
        moved = False

        if st.session_state.state == "TOP":
            if angle_s <= angle_down and st.session_state.frames_in_state >= min_frames:
                st.session_state.state = "BOTTOM"
                st.session_state.frames_in_state = 0
                moved = True

        elif st.session_state.state == "BOTTOM":
            if angle_s >= angle_up and st.session_state.frames_in_state >= min_frames:
                if (not use_hip) or hip_move_ok:
                    st.session_state.reps_total += 1
                    st.session_state.reps_current_set += 1

                    now = time.time()
                    if st.session_state.last_rep_time is not None:
                        st.session_state.rep_times.append(now - st.session_state.last_rep_time)
                    st.session_state.last_rep_time = now

                    # fechou série?
                    if st.session_state.reps_current_set >= target_reps:
                        st.session_state.sets_total += 1
                        st.session_state.reps_per_set.append(st.session_state.reps_current_set)
                        st.session_state.reps_current_set = 0

                st.session_state.state = "TOP"
                st.session_state.frames_in_state = 0
                moved = True

        if moved:
            st.session_state.last_movement_time = time.time()

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        angle = None
        hip_ok = True

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # desenhar (leve)
            mp_drawing.draw_landmarks(
                img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            sx, sy = to_px(r_sh, w, h); ex, ey = to_px(r_el, w, h); wx, wy = to_px(r_wr, w, h)

            angle = calc_angle((sx, sy), (ex, ey), (wx, wy))
            st.session_state.angle_hist.append(angle)
            angle_s = float(np.mean(st.session_state.angle_hist)) if st.session_state.angle_hist else angle
            st.session_state.angle_display = angle_s

            # hip check
            if use_hip:
                r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                _, hip_y = to_px(r_hip, w, h)
                hip_y_norm = hip_y / float(h)
                st.session_state.hip_y_hist.append(hip_y_norm)
                if len(st.session_state.hip_y_hist) > 5:
                    recent = list(st.session_state.hip_y_hist)[-5:]
                    amp = max(recent) - min(recent)
                    hip_ok = amp >= hip_min_delta
                st.session_state.hip_move_ok = hip_ok

            # lógica de contagem
            if angle_s is not None:
                self._count_logic(angle_s, hip_ok)

            # overlay leve
            cv2.rectangle(img, (0,0), (w, 40), (0,0,0), -1)
            cv2.putText(img, f"Reps {st.session_state.reps_total}  |  Serie {st.session_state.sets_total+1} ({st.session_state.reps_current_set}/{target_reps})",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if angle is not None:
                cv2.putText(img, f"Ang: {int(angle)}", (w-110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if use_hip:
                cv2.putText(img, f"Quadril: {'OK' if hip_ok else 'NAO'}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255) if hip_ok else (0,0,255), 2)

        # idle → fecha série corrente
        idle = time.time() - st.session_state.last_movement_time
        if idle >= idle_secs and st.session_state.reps_current_set > 0:
            st.session_state.sets_total += 1
            st.session_state.reps_per_set.append(st.session_state.reps_current_set)
            st.session_state.reps_current_set = 0
            st.session_state.last_movement_time = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------- WebRTC start -------------
webrtc_streamer(
    key="pushup",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=PushUpProcessor,
)

# ------------- Painel de status -------------
st.subheader("Status")
col_a, col_b = st.columns(2)
col_a.write(f"Ângulo (suavizado): **{int(st.session_state.angle_display or 0)}°**")
if use_hip:
    col_b.write(f"Quadril: **{'OK' if st.session_state.hip_move_ok else 'NÃO'}**")

idle_now = int(time.time() - st.session_state.last_movement_time)
st.caption(f"Ocioso: {idle_now}s")

# ------------- Salvar sessão -------------
def finalize_and_save():
    session_end = time.time()
    # fecha série pendente
    if st.session_state.reps_current_set > 0:
        st.session_state.sets_total += 1
        st.session_state.reps_per_set.append(st.session_state.reps_current_set)
        st.session_state.reps_current_set = 0
    avg_rep = (sum(st.session_state.rep_times)/len(st.session_state.rep_times)) if st.session_state.rep_times else None
    summary = SessionSummary(
        start_time=fmt_time(st.session_state.session_start),
        end_time=fmt_time(session_end),
        duration_sec=session_end - st.session_state.session_start,
        total_reps=st.session_state.reps_total,
        total_sets=st.session_state.sets_total,
        reps_per_set=st.session_state.reps_per_set,
        avg_sec_per_rep=avg_rep,
        params={
            "ANGLE_UP_THRESHOLD": angle_up,
            "ANGLE_DOWN_THRESHOLD": angle_down,
            "MIN_FRAMES_IN_STATE": min_frames,
            "SMOOTHING_WINDOW": smooth_win,
            "USE_HIP_CHECK": use_hip,
            "HIP_MIN_DELTA": hip_min_delta,
            "DEFAULT_TARGET_REPS": target_reps,
            "INACTIVITY_SECONDS": idle_secs,
        }
    )
    save_session(summary)
    st.success("Resumo da sessão salvo em `data/sessions/` ✅")
    st.json(asdict(summary))

st.button("💾 Salvar sessão agora", on_click=finalize_and_save)