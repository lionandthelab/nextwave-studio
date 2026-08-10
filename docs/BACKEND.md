# BACKEND — Workcell 서버 · 다중 사용자 · 오프라인 계약

> Phase 12+에서 추가된 서버 계층의 규범 문서. 개체 스키마의 단일 진실은
> `src/schema/entities.ts`(클라이언트/서버 공유 zod)이며, 이 문서는 그 위의
> HTTP 계약·인증·배포·오프라인 정책을 정의한다.

---

## 1. 원칙

1. **서버는 선택적이다.** 앱은 서버 없이도 부팅되고 IndexedDB 로컬 모드로 동작한다
   (기존 정적 호스팅 불변식 유지). 서버가 있으면 다중 사용자·공유·실행 기록이 켜진다.
   클라이언트는 `GET /api/v1/health` 실패 시 로컬 모드로 자연 강등된다.
2. **현장 1차 사용자는 로봇 설치기사다.** 배포는 단일 프로세스(정적 번들 + API 동시
   서빙, 포트 1개), DB는 단일 파일(SQLite), 로그인은 사용자 타일 + 숫자 PIN(공유 단말·
   장갑 전제, 터치 타깃 ≥44px)이다. IT 담당자 없이 켜지는 것이 목표다.
3. **서버는 씬을 해석하지 않는다.** scene/sequence는 봉투(z.unknown())로만 검증·저장
   한다. 도메인 검증(sceneSpecSchema 등)은 클라이언트 로드 시점의 몫이다(§2.9 원칙).
4. **파괴는 언제나 되돌릴 수 있다.** 삭제는 soft-delete(휴지통 30일) + restore API.
   Run은 append-only로 삭제 API가 없다.
5. **동시 편집은 낙관적 버전 + 조언적 잠금.** 저장은 baseVersion 검사(불일치 409 +
   서버 현재본 반환), 편집 중 표시는 TTL 90초 잠금(heartbeat 갱신, 강탈 없음).

## 2. 스택

| 역할 | 선택 | 이유 |
|---|---|---|
| HTTP | Fastify 5 | 스키마 훅·에러 처리 견고, 순수 JS(네이티브 빌드 없음) |
| DB | better-sqlite3 | 단일 파일, 동기 API, win32 프리빌드 — 현장 PC 설치 부담 0 |
| 정적 서빙 | @fastify/static | dist/를 같은 포트에서 서빙(단일 프로세스 배포) |
| 해시 | node:crypto scrypt | PIN 해시(솔트 포함), 의존성 0 |
| 실행 | tsx | TS 직접 실행(dev/prod 동일) |

- 포트: `WORKCELL_PORT`(기본 **8787**). DB 경로: `WORKCELL_DATA_DIR`(기본 `server/data/`).
- dev에서는 Vite 프록시(`/api` → `localhost:8787`)로 CORS를 원천 제거한다.
- Docker: `server` 서비스(권장 전체 배포). 기존 `app`(nginx 정적 전용)은 로컬 모드용으로 유지.

## 3. 인증 · 다중 사용자

- **부트스트랩**: 사용자 0명이면 `GET /auth/bootstrap → { needsSetup: true }` → 설정
  마법사에서 관리자 생성(`POST /auth/setup`, 그때만 열려 있다).
- **로그인**: `GET /auth/users`(타일: id·name·role만) → 타일 선택 → PIN 패드 →
  `POST /auth/login`. 실패 5회 → 60초 잠금(423). PIN은 scrypt(N=16384, 사용자별 솔트).
- **세션**: 토큰 32바이트 무작위(base64url). 서버는 SHA-256 해시만 저장. TTL 30일
  슬라이딩. 클라이언트는 `Authorization: Bearer <token>`. localStorage `workcell.session`.
- **역할**: `admin`(사용자 관리, 완전 삭제, 설정) / `tech`(개체 CRUD). 문서는 사이트
  공유가 기본(팀 모델) — 소유권 대신 **감사**(created_by/updated_by/deleted_by)로 추적.
- **빠른 사용자 전환**: 로그아웃 없이 타일로 복귀(공유 단말) — 세션은 사용자별.

## 4. API (전 경로 `/api/v1`, JSON)

인증 불필요: `health`, `auth/bootstrap`, `auth/users`, `auth/setup`, `auth/login`.
그 외 전부 Bearer 토큰 필수(401). admin 전용은 (A) 표기. 오류 봉투:
`{ error: string, messageKo: string }` + 적절한 상태 코드.

| 메서드·경로 | 요청 → 응답 |
|---|---|
| GET `/health` | → `{ ok, name, version, uptimeSec }` |
| GET `/auth/bootstrap` | → `{ needsSetup, serverName }` |
| POST `/auth/setup` | `{ name, pin }` → `{ token, user }` (사용자 0명일 때만, admin 생성, PIN 6자리+) |
| GET `/auth/users` | → `{ users: UserInfo[] }` (active만) |
| POST `/auth/login` | `{ userId, pin }` → `{ token, user }` · 401 · 423 `{ retryAfterSec }` |
| POST `/auth/logout` | → `{ ok }` (토큰 폐기) |
| GET `/auth/me` | → `{ user }` |
| GET `/users` (A) | → `{ users }` (inactive 포함) |
| POST `/users` (A) | `{ name, pin, role }` → `{ user }` |
| PATCH `/users/:id` | `{ name?, pin?, role?, active? }` → `{ user }` (본인 pin 변경은 비-admin 허용) |

**개체 CRUD** — `E ∈ { processes, tasks, blocks, devices }`, 문서 스키마는 entities.ts:

| 메서드·경로 | 요청 → 응답 |
|---|---|
| GET `/E` | `?q=&includeDeleted=0&processId=(tasks만)` → `{ items: EntityMeta[] }` (payload 없는 메타, updatedAt 내림차순. 조회 시 30일 지난 휴지통 행 지연 완전삭제) |
| POST `/E` | `SaveRequest{ doc, baseVersion: null }` → 201 `RecordEnvelope` (id는 클라이언트 발급 uuid — 오프라인 생성 지원. 중복 id는 409) |
| GET `/E/:id` | → `RecordEnvelope` (휴지통 행도 반환하되 meta.deletedAtIso로 표시) |
| PUT `/E/:id` | `SaveRequest{ doc, baseVersion }` → `RecordEnvelope` · 409 `ConflictResponse`(현재본 포함) |
| DELETE `/E/:id` | → `{ restoreUntilIso }` (soft — admin이 아니어도 가능, 완전 삭제는 없음) |
| POST `/E/:id/restore` | → `RecordEnvelope` |

**잠금 · 실행 기록 · 통계**:

| 메서드·경로 | 요청 → 응답 |
|---|---|
| POST `/locks/:kind/:id` | `{ action: 'acquire'\|'heartbeat'\|'release' }` → `{ lock: LockInfo \| null }` · 423 `{ lock }`(타인 보유) |
| GET `/locks/:kind/:id` | → `{ lock: LockInfo \| null }` |
| POST `/runs` | `RunRecord` → 201 `{ id }` (append-only, operator는 토큰에서 강제) |
| GET `/runs` | `?taskId=&limit=50&offset=0` → `{ items: RunRecord[], total }` (startedAt 내림차순) |
| GET `/runs/:id` | → `RunRecord` |
| GET `/tasks/:id/stats` | → `TaskStats` |

## 5. DB 스키마 (SQLite, WAL 모드)

```
users(id PK, name, role, pin_hash, salt, active, failed_attempts, locked_until, created_at)
sessions(token_hash PK, user_id→users, created_at, expires_at, last_seen_at)
entities(kind, id, name, process_id NULL, payload JSON, version,
         created_at, created_by, updated_at, updated_by,
         deleted_at NULL, deleted_by NULL, PRIMARY KEY(kind, id))
locks(kind, id, user_id, acquired_at, expires_at, PRIMARY KEY(kind, id))
runs(id PK, task_id, payload JSON, started_at, operator_id)   -- append-only
```

- 4개 문서 개체는 `entities` 단일 테이블(kind 구분) — 마이그레이션 표면 최소화.
- 목록 API는 payload에서 요약(taskSummary 등)만 추출해 EntityMeta로 돌려준다.
- 스키마 변경은 `migrations` 배열(버전 정수 + SQL)로 전진만 한다.

## 6. 오프라인 · 동기화 (클라이언트 `src/api/`)

- `ApiClient`가 연결 상태를 소유한다: health 폴링 + fetch 실패 감지 →
  `'online' | 'offline' | 'local-only'`(서버 미설정) 3상태. UI는 상단 배지로 항상 표시.
- **읽기**: 성공 응답을 IndexedDB 캐시(`workcell-api-cache`)에 기록, 오프라인이면
  캐시 서빙 + "오프라인 — 마지막 동기화 n분 전" 배너.
- **쓰기**: 오프라인이면 outbox 큐(IndexedDB)에 적재 후 재연결 시 순서대로 재전송.
  409 충돌은 **자동 병합하지 않는다** — 항목에 충돌 배지를 남기고 사용자가
  "서버본 열기 / 내 것으로 덮어쓰기 / 사본으로 저장"을 고른다(수작업 존중 — §2.11 정신).
- 로그인 불가(오프라인) 시 **로컬 모드로 계속**을 제공한다 — 작업물은 기존 IndexedDB
  문서 저장소에 남고, 재연결 후 서버로 올릴 수 있다.

## 7. 검증 게이트

- 서버 단위 테스트(vitest, `server/**/*.test.ts`): 인증(성공·실패·잠금·세션 만료),
  CRUD 왕복, 버전 충돌 409, soft-delete/restore, 잠금 TTL, runs append·통계.
- `npm run verify`가 서버 typecheck(`tsconfig.server.json`)·lint·테스트를 포함한다.
- 통합 게이트(추후): `--expect=multi-user` — 두 세션이 같은 작업을 열어 잠금 표시·
  충돌 해결 UI까지 브라우저로 검증.
