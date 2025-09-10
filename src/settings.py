# Parâmetros ajustáveis em um lugar só

# Limiar de ângulos (graus) para estados do cotovelo
ANGLE_UP_THRESHOLD = 160      # "CIMA" (braço estendido)
ANGLE_DOWN_THRESHOLD = 70     # "BAIXO" (braço flexionado)

# Robustez do contador
MIN_FRAMES_IN_STATE = 3       # debouncing (frames mínimos por estado)
SMOOTHING_WINDOW = 5          # média móvel do ângulo

# Anti-burla: quadril também precisa se mover um pouco
USE_HIP_CHECK = True
HIP_MIN_DELTA = 0.03          # proporção da altura do frame

# Séries
DEFAULT_TARGET_REPS = 12      # reps alvo por série
INACTIVITY_SECONDS = 5.0      # pausa automática se ficar parado

# Vídeo
CAM_INDEX = 0                 # índice da webcam
DRAW_SKELETON = True          # desenhar landmarks

# Áudio
BEEP_ON_REP = True            # beep a cada repetição
BEEP_ON_SET = True            # beep ao concluir série