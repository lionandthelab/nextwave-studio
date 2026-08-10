# syntax=docker/dockerfile:1
#
# robot-sim-web — 크로스 플랫폼 패키징 (Windows / Linux / macOS 동일 동작)
#
# 이 앱은 백엔드가 없다(PRD NFR-1). 컨테이너가 하는 일은 정적 번들을 만들어
# 서빙하는 것뿐이므로, 호스트 OS·Node 버전·툴체인에 상관없이 결과가 같다.
#
# 타깃(빌드 스테이지):
#   runtime  (기본) 프로덕션 — nginx로 dist/ 서빙
#   dev              개발 서버 — Vite HMR (docker-compose의 dev 프로파일)
#   verify           검증 — typecheck + lint + 단위 테스트 (CI용)
#
# 사용:
#   docker build -t robot-sim-web .            # runtime 이미지
#   docker build --target verify .             # 검증만 수행
#   docker run --rm -p 8080:80 robot-sim-web   # http://localhost:8080

# ── 공통 의존성 레이어 ───────────────────────────────────────────────
# package*.json만 먼저 복사해 소스 변경 시 npm ci 레이어가 캐시되게 한다.
FROM node:22-alpine AS deps
WORKDIR /app

# playwright(devDependency)의 postinstall 브라우저 다운로드를 막는다.
# 브라우저 게이트는 컨테이너 밖(호스트)에서 실행한다 — README/USAGE 참조.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

COPY package.json package-lock.json ./
RUN npm ci

# ── 검증 스테이지 (CI: docker build --target verify .) ───────────────
# tsc --noEmit + eslint + vitest. 실패하면 빌드가 여기서 멈춘다.
FROM deps AS verify
COPY . .
RUN npm run verify

# ── 빌드 스테이지 ────────────────────────────────────────────────────
FROM deps AS build
COPY . .
RUN npm run build        # tsc --noEmit && vite build → /app/dist

# ── 개발 스테이지 (docker compose --profile dev up) ──────────────────
# 소스는 bind mount로 주입된다. --host 0.0.0.0으로 컨테이너 밖에서 접속 가능하게.
FROM deps AS dev
WORKDIR /app
ENV VITE_USE_POLLING=true
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ── 서버 스테이지 (전체 배포: 정적 번들 + API 단일 프로세스) ─────────
# docs/BACKEND.md §2 — 프로세스 1개 · 포트 1개(8787) · SQLite 파일 1개.
# deps 레이어(alpine/musl)를 재사용하지 않고 slim(glibc)에서 npm ci를 다시 하는 이유:
# better-sqlite3 프리빌드 바이너리가 glibc 대상이라 musl에서는 소스 빌드가 필요해진다.
FROM node:22-slim AS server
WORKDIR /app

ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# 서버가 dist/를 같은 포트에서 서빙한다 — typecheck는 verify 스테이지 몫이므로 빌드만
RUN npx vite build

ENV NODE_ENV=production
ENV WORKCELL_PORT=8787
EXPOSE 8787

# DB는 /app/server/data (compose가 호스트 볼륨을 마운트한다)
CMD ["npx", "tsx", "server/index.ts"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://localhost:8787/api/v1/health').then((r)=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

# ── 런타임 스테이지 (기본) ───────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

LABEL org.opencontainers.image.title="robot-sim-web" \
      org.opencontainers.image.description="브라우저 완결형 로봇 시뮬레이터 (Rapier + three.js)" \
      org.opencontainers.image.source="https://github.com/lionandthelab/nextwave-studio"

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# busybox wget — alpine 기본 포함(curl 불필요)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --spider -q http://localhost/ || exit 1
