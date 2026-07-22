# robot-sim-web

브라우저에서 완결되는 **간소화된 IsaacSim형 로봇 시뮬레이터**. 로봇/사물 오브젝트를
드래그앤드롭으로 배치해 가상 환경을 구성하고, **자연어 문장으로 로봇 제어 액션 플로우를
생성**하며(n8n형 노드 그래프로 편집), 시뮬레이터에 동작을 하나씩 요청해 재생하고,
로봇–사물 충돌을 실시간 감지한다. 백엔드 없이 정적 호스팅으로 배포하는 것을 목표로 한다.

## 스택

| 역할 | 기술 |
|-----|------|
| 물리 엔진 | Rapier (`@dimforge/rapier3d-compat`, WASM) |
| 렌더링 | three.js |
| 로봇 모델 로딩 | `urdf-loader` (URDF → three.js) |
| 빌드/개발 | Vite + TypeScript (strict) |
| 업그레이드 경로 | 접촉 물리 정밀도 필요 시 물리 계층만 MuJoCo WASM(`@mujoco/mujoco`)으로 교체 |

## 핵심 아이디어

- **물리가 진실**: 트랜스폼의 단일 원천은 Rapier. three.js는 물리 결과를 비추는 거울.
- **데이터 주도**: 씬(`SceneSpec`)과 제어(`ControlSequence`)는 선언적 JSON. 엔진은 해석기.
- **자연어 → 플로우**: 문장을 씬에 그라운딩해 액션 플로우 초안 생성. 노드 그래프는
  ControlSequence의 무손실 뷰. 사람이 검토·수정 후 Play해야 실행(human-in-the-loop).
- **직접 조작**: 라이브러리에서 물체/로봇을 드래그해 추가, 3D 파일 임포트, 치수/노드 순서를
  드래그로 편집. 실행 시 활성 노드(그래프)와 로봇 동작(뷰포트)이 동기 강조.
- **결정론 지향**: 고정 timestep + accumulator로 프레임레이트와 물리를 분리.
- **계층 격리**: 물리 엔진 의존을 한 계층에 가둬 MuJoCo로 교체 가능.

## 문서 (읽는 순서)

1. `CLAUDE.md` — 프로젝트 헌법(불변식·컨벤션·워크플로). Claude Code가 먼저 읽는 파일
2. `docs/PRD.md` — 목표·범위·요구사항·non-goals
3. `docs/ARCHITECTURE.md` — 계층 구조·데이터 흐름·모듈 경계
4. `docs/DATA_MODEL.md` — SceneSpec / EntitySpec / ControlSequence 스키마
5. `docs/SIMULATION.md` — 시뮬 루프·제어 player·충돌 감지(Rapier API)
6. `docs/UX_DESIGN.md` — 화면 UX 설계서(워크스페이스·n8n형 노드그래프·씬빌더·실행)
7. `docs/PLANNER.md` — 자연어 → ControlSequence 생성·검증·복구
8. `docs/ROADMAP.md` — 단계별 마일스톤 + 검증 게이트
9. `AGENTS.md` — 리뷰 역할 / `EXPERIMENTS.md` — 결정 로그

## 빠른 시작 (Phase 0 기준)

```bash
npm install
npm run dev        # Vite 개발 서버
```

> 아직 구현은 Phase 0(스캐폴딩)부터 시작한다. `docs/ROADMAP.md`의 Phase 순서와
> 검증 게이트를 따른다. 물리 API 사용 전 `await RAPIER.init()`가 반드시 선행되어야 한다.

## 디렉터리

```
CLAUDE.md · AGENTS.md · EXPERIMENTS.md · README.md
package.json · vite.config.ts
docs/            설계문서
.claude/commands/ Claude Code 커스텀 커맨드
src/             구현 (core / render / schema / ui / assets)
```

전체 구조와 계층 규칙은 `CLAUDE.md` §3 참조.

## 주의

이 시뮬레이터는 교육·프로토타입·데모용이다. 물리 정확도 인증이 필요한 안전 검증
용도로는 적합하지 않다(`docs/PRD.md` §6/§8).
