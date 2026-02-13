# AutoGrip-Sim Engine - Architecture Reference

> **Version**: 0.1.0
> **Last Updated**: 2025-01
>
> 이 문서 하나로 프로젝트 전체 구조, 데이터 모델, API 계약, 모듈 관계를 파악할 수 있습니다.

---

## 1. Project Structure

```
autogrip-sim/
├── docker-compose.yml              # 3 services: backend, isaac-sim, backend-dev
├── docs/
│   └── architecture.md             # ← 이 문서
│
└── backend/
    ├── run.py                      # uvicorn dev server launcher
    ├── requirements.txt            # Python dependencies
    ├── .env.example                # Environment variable template
    │
    ├── app/
    │   ├── main.py                 # FastAPI entry point, lifespan, CORS, static mount
    │   ├── config.py               # pydantic-settings 기반 설정 (Settings)
    │   ├── models.py               # Pydantic request/response schemas
    │   ├── session_manager.py      # Async-safe singleton session store
    │   │
    │   ├── api/v1/
    │   │   ├── upload.py           # POST /upload/cad, /upload/manual
    │   │   ├── generate.py         # POST /generate/start, /stop, GET /status, /code
    │   │   └── monitor.py          # GET /monitor/stream (SSE), /logs, /result GIF
    │   │
    │   ├── core/
    │   │   ├── llm_engine.py       # RAG 기반 코드 생성 (LangChain + ChromaDB + OpenAI)
    │   │   ├── parser.py           # PDF 매뉴얼 파서 (PyMuPDF)
    │   │   └── loop_controller.py  # 대안 루프 오케스트레이터 (LoopResult + GIF 저장)
    │   │
    │   └── sim_interface/
    │       ├── connector.py        # IsaacSimConnector + MockSimulator
    │       ├── validator.py        # 4-check 물리 검증 엔진
    │       └── runner.py           # generate.py ↔ connector+validator 브릿지
    │
    ├── static/
    │   ├── index.html              # 4-panel 대시보드 (Settings, Viewer, Code, Log)
    │   ├── css/style.css           # Dark theme CSS (CSS Grid)
    │   └── js/
    │       ├── app.js              # AutoGripApp 메인 로직 + SSE 핸들러
    │       └── three-viewer.js     # STL 3D 프리뷰 (Canvas 2D)
    │
    ├── tests/
    │   ├── conftest.py             # 공유 fixtures (AsyncClient, STL/PDF 생성)
    │   ├── test_upload.py
    │   ├── test_session.py
    │   ├── test_validator.py
    │   ├── test_connector.py
    │   ├── test_parser.py
    │   └── test_generate.py
    │
    └── docker/
        ├── Dockerfile              # Python 3.11-slim backend
        ├── Dockerfile.isaac-sim    # nvcr.io/nvidia/isaac-sim:4.2.0 기반
        └── sim_scripts/
            └── sim_server.py       # Isaac Sim 내부 REST API
```

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Vanilla JS)                        │
│  index.html + app.js + three-viewer.js + style.css                 │
│                                                                     │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │ Settings │ │ 3D Viewer    │ │ Code     │ │ Log Panel         │  │
│  │ Panel    │ │ (STL/GIF)    │ │ Panel    │ │ (SSE streaming)   │  │
│  └────┬─────┘ └──────────────┘ └──────────┘ └────────┬──────────┘  │
│       │          REST API                       SSE   │             │
└───────┼───────────────────────────────────────────────┼─────────────┘
        │                                               │
        ▼                                               ▼
┌───────────────────────── FastAPI Backend ──────────────────────────┐
│                                                                    │
│  /api/v1/upload/*    → upload.py      (파일 저장 + trimesh 분석)   │
│  /api/v1/generate/*  → generate.py    (루프 시작/중지/상태)        │
│  /api/v1/monitor/*   → monitor.py     (SSE 스트림 + 로그 + GIF)   │
│  /api/v1/sessions    → main.py        (세션 생성)                  │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │SessionManager│  │ LLM Engine   │  │   Sim Interface        │   │
│  │ (in-memory)  │  │ (LangChain)  │  │ connector + validator  │   │
│  └──────────────┘  └──────┬───────┘  └───────────┬────────────┘   │
│                           │                       │                │
└───────────────────────────┼───────────────────────┼────────────────┘
                            │                       │
                   ┌────────▼────────┐    ┌─────────▼─────────┐
                   │  OpenAI API     │    │  Isaac Sim         │
                   │  (GPT-4o)       │    │  (Docker/Mock)     │
                   │  + ChromaDB     │    │                    │
                   └─────────────────┘    └────────────────────┘
```

---

## 3. Self-Correcting Loop (핵심 플로우)

```
              ┌──────────────────┐
              │  Upload CAD/PDF  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  POST /start     │  → GenerateStartRequest
              └────────┬─────────┘
                       ▼
         ┌─── iteration = 1 ─────────────────┐
         │                                     │
         │  ┌───────────────────────────────┐  │
         │  │ Step 1: LLM Code Generation   │  │
         │  │  - iteration 1: generate_code │  │
         │  │  - iteration N: refine_code   │  │
         │  └──────────────┬────────────────┘  │
         │                 ▼                    │
         │  ┌───────────────────────────────┐  │
         │  │ Step 2: Simulation            │  │
         │  │  - MockSimulator (개발)       │  │
         │  │  - Isaac Sim Docker (프로덕션) │  │
         │  └──────────────┬────────────────┘  │
         │                 ▼                    │
         │  ┌───────────────────────────────┐  │
         │  │ Step 3: Validation (4 checks) │  │
         │  │  ✓ Hold Test (z ≥ 0.1m)      │  │
         │  │  ✓ Contact Test (F ≥ 0.5N)   │  │
         │  │  ✓ Stability (ω ≤ 1.0 rad/s) │  │
         │  │  ✓ Force (F ≤ 500N)          │  │
         │  └──────────────┬────────────────┘  │
         │                 ▼                    │
         │  ┌───────────────────────────────┐  │
         │  │ Step 4: Termination Check     │  │
         │  │  - 3 consecutive passes?  ──YES──── ✅ SUCCESS
         │  │  - max_iterations?        ──YES──── ❌ FAILED
         │  │  - else: classify error,      │  │
         │  │    apply correction strategy  │  │
         │  └──────────────┬────────────────┘  │
         │                 │ RETRY              │
         └─────────────────┘                    │
                                                │
```

---

## 4. Data Models

### 4.1 API Request/Response Models (`app/models.py`)

| Model | Fields | Description |
|-------|--------|-------------|
| **UploadResponse** | `id`, `filename`, `file_type`, `size_bytes` | 파일 업로드 응답 |
| **CADMetadata** | `filename`, `format`, `dimensions{x,y,z}`, `volume?`, `center_of_mass?` | trimesh로 추출한 CAD 메타데이터 |
| **SessionCreate** | `cad_file_id`, `manual_file_id?`, `robot_model="unitree_h1_hand"` | 세션 생성 요청 |
| **SessionResponse** | `session_id`, `status`, `created_at` | 세션 생성 응답 |
| **GenerateStartRequest** | `cad_file_id`, `manual_file_id?`, `robot_model`, `max_iterations=20`, `success_threshold=3`, `session_id?` | 루프 시작 요청 (세션 자동 생성) |
| **SimulationResult** | `iteration`, `success`, `checks{name→bool}`, `error_log?`, `code_diff?` | 1회 시뮬레이션 결과 |
| **LoopStatus** | `session_id`, `current_iteration`, `max_iterations`, `status`, `results[]`, `final_code?` | 전체 루프 상태 |

### 4.2 Internal Data Models

#### `_SessionData` (session_manager.py)
```python
session_id: str
cad_file_id: str
manual_file_id: str | None
robot_model: str
status: str                        # "created" | "running" | "success" | "failed" | "stopped"
created_at: str                    # ISO 8601 UTC
current_iteration: int
max_iterations: int                # default 20
success_threshold: int             # default 3
results: list[SimulationResult]
logs: list[dict]                   # {timestamp, level, message, data}
generated_code: str | None
task: asyncio.Task | None          # Background correction loop task
```

#### `CheckResult` (validator.py)
```python
name: str          # "hold_test" | "contact_test" | "stability_test" | "force_test"
passed: bool
value: float       # 측정값
threshold: float   # 기준값
message: str       # 상세 메시지
```

#### `ValidationResult` (validator.py)
```python
success: bool                      # all checks passed
checks: dict[str, CheckResult]     # 4개 체크 결과
error_log: str                     # 실패 시 에러 메시지
suggestions: list[str]             # LLM에 전달할 개선 제안
```

#### `SimulationContext` (connector.py)
```python
running: bool
headless: bool
time_step: float                   # 1/120 s
elapsed_time: float
frame_count: int
gravity: float                     # -9.81 m/s^2
objects: dict[str, ObjectState]
robot: RobotState | None
ground_plane: bool
frames: list[bytes]                # PNG frame bytes
logs: list[str]
```

#### `ManualData` (parser.py)
```python
raw_text: str                      # 전체 PDF 텍스트
chunks: list[str]                  # 임베딩용 청크 (1000자, 200자 오버랩)
joint_names: list[str]             # regex 추출된 관절명
control_functions: list[str]       # 제어 함수명
motor_specs: dict[str, dict]       # torque, speed, joint_range, payload, grip_force
```

---

## 5. API Endpoints

### 5.1 File Upload

| Method | Path | Request | Response | Description |
|--------|------|---------|----------|-------------|
| POST | `/api/v1/upload/cad` | `multipart/form-data` (file) | `UploadResponse` | STL/OBJ/STEP 업로드, trimesh 메타데이터 추출 |
| POST | `/api/v1/upload/manual` | `multipart/form-data` (file) | `UploadResponse` | PDF 매뉴얼 업로드 |
| GET | `/api/v1/upload/{file_id}` | - | `dict` | 업로드된 파일 메타데이터 조회 |

### 5.2 Session Management

| Method | Path | Request | Response | Description |
|--------|------|---------|----------|-------------|
| POST | `/api/v1/sessions` | `SessionCreate` | `SessionResponse` | 세션 생성 |

### 5.3 Code Generation Loop

| Method | Path | Request | Response | Description |
|--------|------|---------|----------|-------------|
| POST | `/api/v1/generate/start` | `GenerateStartRequest` | `LoopStatus` | 루프 시작 (세션 자동 생성) |
| POST | `/api/v1/generate/stop/{session_id}` | - | `LoopStatus` | 루프 중지 |
| GET | `/api/v1/generate/status/{session_id}` | - | `LoopStatus` | 현재 상태 조회 |
| GET | `/api/v1/generate/code/{session_id}` | - | `{session_id, code}` | 최신 생성 코드 조회 |

### 5.4 Monitoring (SSE)

| Method | Path | Response | Description |
|--------|------|----------|-------------|
| GET | `/api/v1/monitor/stream/{session_id}` | `text/event-stream` | 실시간 이벤트 스트림 |
| GET | `/api/v1/monitor/logs/{session_id}` | `list[dict]` | 전체 로그 히스토리 |
| GET | `/api/v1/monitor/result/{session_id}/gif` | `image/gif` | 결과 GIF 다운로드 |

#### SSE Event Types

| Event Name | Data Format | Description |
|------------|-------------|-------------|
| `log` | `{timestamp, level, message, data}` | 로그 엔트리 |
| `iteration_start` | `{iteration}` | 새 이터레이션 시작 |
| `iteration_result` | `SimulationResult` (JSON) | 이터레이션 결과 |
| `code_update` | `{code}` | 생성 코드 변경 |
| `complete` | `{status, gif_url, code, ...LoopStatus}` | 루프 완료 (성공/실패/중지) |
| `error` | `{message}` | 에러 발생 |

---

## 6. Module Dependencies

```
main.py
  ├── config.py (settings)
  ├── models.py (Pydantic schemas)
  ├── session_manager.py (SessionManager singleton)
  │
  ├── api/v1/upload.py
  │     ├── config.py
  │     ├── models.py (CADMetadata, UploadResponse)
  │     ├── session_manager.py
  │     └── [trimesh] (CAD 파싱)
  │
  ├── api/v1/generate.py
  │     ├── config.py
  │     ├── models.py (LoopStatus, SimulationResult)
  │     ├── session_manager.py
  │     ├── core/llm_engine.py (lazy import)
  │     └── sim_interface/runner.py (lazy import)
  │
  └── api/v1/monitor.py
        ├── config.py
        └── session_manager.py

core/llm_engine.py
  ├── config.py
  ├── core/parser.py (ManualParser)
  ├── [langchain-openai] (ChatOpenAI, OpenAIEmbeddings)
  └── [langchain-chroma] (Chroma vector store)

core/parser.py
  └── [pymupdf/fitz] (PDF 텍스트 추출)

sim_interface/runner.py
  ├── sim_interface/connector.py (IsaacSimConnector)
  └── sim_interface/validator.py (GraspValidator)

sim_interface/connector.py
  ├── config.py
  └── MockSimulator (내장, 개발용)

sim_interface/validator.py
  └── (standalone, 외부 의존성 없음)
```

---

## 7. Validation Thresholds

| Check | Constant | Default | Unit | Pass Condition |
|-------|----------|---------|------|----------------|
| Hold Test - Height | `HOLD_HEIGHT_THRESHOLD` | 0.1 | m | `obj_z ≥ threshold` |
| Hold Test - Duration | `HOLD_DURATION_THRESHOLD` | 5.0 | s | `duration ≥ threshold` |
| Contact Test | `MIN_CONTACT_FORCE` | 0.5 | N | `total_force ≥ threshold` |
| Stability Test | `MAX_ANGULAR_VELOCITY` | 1.0 | rad/s | `ω_magnitude ≤ threshold` |
| Force Test | `MAX_SAFE_FORCE` | 500.0 | N | `max(torque, contact) ≤ threshold` |

---

## 8. Error Classification & Correction Strategies

LLM 코드 수정 시 에러 타입에 따라 자동으로 적용되는 전략:

| Error Type | Detection Keywords | Correction Strategy |
|------------|-------------------|---------------------|
| `slip` | slip, dropped, lost grip, fell | torque +30-50%, 그립 폭 축소, pre-grasp squeeze 추가 |
| `collision` | collision, collide, penetration | approach clearance +0.05m, 접근 각도 조정, 속도 -30% |
| `no_contact` | no contact, miss, not touching | 그립 위치를 COM 쪽으로, 그립 폭 확대, 접근 높이 -0.02m |
| `timeout` | timeout, timed out, too slow | waypoint 축소, 속도 +20%, 불필요 대기 제거 |
| `unknown` | (default) | joint 경로 확인, physics step 검증, error handling 추가 |

---

## 9. Configuration Reference

### Environment Variables (`config.py` → `Settings`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | OpenAI API 키 |
| `LLM_MODEL` | `"gpt-4o"` | LLM 모델명 |
| `LLM_TEMPERATURE` | `0.2` | LLM 생성 온도 |
| `LLM_MAX_TOKENS` | `4096` | LLM 최대 토큰 |
| `ISAAC_SIM_PATH` | `"/isaac-sim"` | Isaac Sim 설치 경로 |
| `ISAAC_SIM_HEADLESS` | `true` | GUI 없이 실행 |
| `ISAAC_SIM_DOCKER_IMAGE` | `"nvcr.io/nvidia/isaac-sim:4.2.0"` | Isaac Sim Docker 이미지 |
| `APP_HOST` | `"0.0.0.0"` | 서버 호스트 |
| `APP_PORT` | `8000` | 서버 포트 |
| `UPLOAD_DIR` | `"./uploads"` | 업로드 디렉토리 |
| `MAX_UPLOAD_SIZE_MB` | `100` | 최대 업로드 크기 (MB) |
| `MAX_LOOP_ITERATIONS` | `20` | 최대 루프 반복 횟수 |
| `SUCCESS_THRESHOLD` | `3` | 연속 성공 횟수 |
| `CHROMA_PERSIST_DIR` | `"./chroma_db"` | ChromaDB 저장 경로 |
| `LOG_LEVEL` | `"INFO"` | 로그 레벨 |

### Derived Paths

| Property | Value |
|----------|-------|
| `upload_path` | `{UPLOAD_DIR}` |
| `cad_upload_path` | `{UPLOAD_DIR}/cad` |
| `manual_upload_path` | `{UPLOAD_DIR}/manuals` |
| `results_path` | `{UPLOAD_DIR}/results` |

---

## 10. Docker Services

| Service | Image | Port | GPU | Profile | Description |
|---------|-------|------|-----|---------|-------------|
| `backend` | `Dockerfile` | 8000 | No | default | FastAPI 프로덕션 서버 |
| `isaac-sim` | `Dockerfile.isaac-sim` | 9090 | Yes (nvidia runtime) | default | Isaac Sim 시뮬레이션 서버 |
| `backend-dev` | `Dockerfile` | 8000 | No | dev | 개발용 (MockSimulator, hot-reload) |

### 실행 방법

```bash
# 개발 (MockSimulator 사용, GPU 불필요)
docker compose --profile dev up backend-dev

# 프로덕션 (Isaac Sim + GPU 필요)
docker compose up backend isaac-sim
```

---

## 11. MockSimulator Behavior

개발 환경에서 Isaac Sim 없이 동작하는 MockSimulator의 특성:

- **코드 품질 분석**: regex로 torque, grasp_width, approach_height, error handling, contact check, hold phase 여부 판단
- **성공 확률 계산**: `base(0.05) + torque_bonus + width_bonus + feature_bonuses + iteration_bonus`
- **실패 모드 결정**: 코드 결함에 따라 slip/collision/no_contact/timeout 중 가중 랜덤 선택
- **이터레이션 보너스**: `min(0.15, iteration * 0.03)` → 후반 이터레이션일수록 성공 확률 상승
- **프레임 생성**: 4x4 PNG (성공=초록, 실패=빨강)

---

## 12. Testing

```bash
# 전체 테스트
cd backend && pytest -v

# 특정 모듈
pytest tests/test_validator.py -v
pytest tests/test_connector.py -v
pytest tests/test_upload.py -v
```

### Test Coverage

| Test File | Module | Test Count | Covers |
|-----------|--------|------------|--------|
| `test_validator.py` | validator.py | 13 | 4 checks, 경계값, 복합 실패 |
| `test_connector.py` | connector.py | - | MockSimulator, 코드 품질 분석 |
| `test_upload.py` | upload.py | - | CAD/PDF 업로드, 확장자 검증 |
| `test_session.py` | session_manager.py | - | CRUD, 동시성, 상태 전이 |
| `test_parser.py` | parser.py | - | PDF 파싱, 관절명/함수 추출 |
| `test_generate.py` | generate.py | - | 루프 시작/중지/상태 조회 |

---

## 13. Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.115.6 | Web framework |
| uvicorn | 0.34.0 | ASGI server |
| langchain | 0.3.14 | LLM orchestration |
| langchain-openai | 0.3.0 | OpenAI integration |
| chromadb | 0.5.23 | Vector store |
| pymupdf | 1.25.1 | PDF parsing |
| trimesh | 4.5.3 | CAD/mesh processing |
| sse-starlette | 2.2.1 | Server-Sent Events |
| pydantic-settings | 2.7.1 | Configuration management |
| imageio | 2.36.1 | GIF generation |
| httpx | 0.28.1 | Async HTTP client + testing |
| pytest-asyncio | 0.25.0 | Async test support |
