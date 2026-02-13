# AutoGrip-Sim Engine

CAD 파일 기반 로봇 파지(Grasping) 코드를 자동 생성하고, NVIDIA Isaac Sim 시뮬레이션으로 검증한 뒤, 실패 시 LLM이 스스로 코드를 수정하는 **Self-Correcting Loop** 시스템입니다.

## Overview

```
Upload CAD → LLM 코드 생성 → Isaac Sim 시뮬레이션 → 물리 검증 (4 checks)
                ↑                                           │
                └───────── 에러 분류 + 자동 수정 ←──────────┘
                         (3회 연속 성공 시 완료)
```

**Target Hardware**: Unitree Z1 Arm + H1 Hand (300만원 예산 범위)
**Target Environment**: Ubuntu 24.04 LTS + NVIDIA RTX 5090

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla JS, HTML5, CSS3 (No Build Tool) |
| Backend | Python FastAPI (Modularized) |
| LLM Engine | LangChain + OpenAI GPT-4o + ChromaDB (RAG) |
| Simulation | NVIDIA Isaac Sim 4.2.0 (Docker) |
| PDF Parsing | PyMuPDF (fitz) |
| CAD Processing | trimesh |
| Real-time | Server-Sent Events (SSE) |
| Container | Docker Compose |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- OpenAI API Key

### 2. Setup

```bash
cd autogrip-sim/backend

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 3. Run (Development)

```bash
# 개발 서버 (MockSimulator 사용, GPU 불필요)
python run.py
```

브라우저에서 `http://localhost:8000` 접속

### 4. Run (Docker)

```bash
cd autogrip-sim

# 개발 모드 (MockSimulator)
docker compose --profile dev up backend-dev

# 프로덕션 (Isaac Sim + NVIDIA GPU 필요)
docker compose up backend isaac-sim
```

## Usage

1. **CAD 파일 업로드**: .stl, .obj, .step 형식의 3D 모델 업로드
2. **로봇 매뉴얼 업로드** (선택): PDF 매뉴얼로 RAG 기반 코드 생성 품질 향상
3. **로봇 모델 선택**: Unitree H1 Hand 등
4. **파라미터 설정**: 최대 반복 횟수, 연속 성공 기준
5. **시작**: 자동으로 코드 생성 → 시뮬레이션 → 검증 → 수정 루프 실행
6. **결과 확인**: 검증된 코드 + 시뮬레이션 GIF 다운로드

## Validation Checks

시뮬레이션 결과는 4가지 물리 기준으로 검증됩니다:

| Check | Condition | Threshold |
|-------|-----------|-----------|
| Hold Test | 오브젝트 Z 높이 유지 + 5초 이상 홀드 | z >= 0.1m, duration >= 5.0s |
| Contact Test | 그리퍼-오브젝트 접촉력 | force >= 0.5N |
| Stability Test | 오브젝트 회전 안정성 | angular_velocity <= 1.0 rad/s |
| Force Test | 안전 범위 내 힘 | force <= 500N |

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload/cad` | CAD 파일 업로드 |
| POST | `/api/v1/upload/manual` | 매뉴얼 PDF 업로드 |
| POST | `/api/v1/generate/start` | 자동 생성 루프 시작 |
| POST | `/api/v1/generate/stop/{id}` | 루프 중지 |
| GET | `/api/v1/generate/status/{id}` | 상태 조회 |
| GET | `/api/v1/monitor/stream/{id}` | SSE 실시간 스트림 |

전체 API 및 데이터 모델은 [docs/architecture.md](docs/architecture.md) 참조.

## Testing

```bash
cd autogrip-sim/backend
pytest -v
```

## Project Structure

```
autogrip-sim/
├── docker-compose.yml
├── README.md
├── docs/architecture.md         # 전체 아키텍처 레퍼런스
└── backend/
    ├── app/
    │   ├── main.py              # FastAPI entry point
    │   ├── config.py            # Settings
    │   ├── models.py            # Pydantic schemas
    │   ├── session_manager.py   # Session store
    │   ├── api/v1/              # REST endpoints
    │   ├── core/                # LLM engine + PDF parser
    │   └── sim_interface/       # Isaac Sim connector + validator
    ├── static/                  # Frontend (HTML/CSS/JS)
    ├── tests/                   # pytest test suite
    └── docker/                  # Dockerfiles
```

상세 구조와 데이터 모델은 [docs/architecture.md](docs/architecture.md) 참조.

## License

Private - Lion and the Lab
