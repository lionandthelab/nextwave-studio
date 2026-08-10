// server/config.ts — 서버 환경 설정 (env → 타입 있는 설정 객체)
//
// 왜 분리하나: 현장 PC에서 IT 담당자 없이 켜지는 것이 목표다(docs/BACKEND.md §1).
// 설정은 env 변수 3개가 전부이고 나머지는 안전한 기본값으로 동작해야 한다.
// process.env를 여러 파일에서 직접 읽으면 기본값이 흩어지므로 이 파일이 단일 진실이다.

import { resolve } from 'node:path';

export interface ServerConfig {
  /** HTTP 포트 — WORKCELL_PORT (기본 8787, docs/BACKEND.md §2) */
  readonly port: number;
  /** SQLite 데이터 디렉터리 — WORKCELL_DATA_DIR (기본 server/data) */
  readonly dataDir: string;
  /** DB 파일 절대 경로 (dataDir/workcell.db) */
  readonly dbPath: string;
  /** 부트스트랩/health에 표시되는 서버 이름 — WORKCELL_SERVER_NAME */
  readonly serverName: string;
}

export const DEFAULT_PORT = 8787;
export const DEFAULT_DATA_DIR = 'server/data';
export const DEFAULT_SERVER_NAME = 'Workcell 서버';

/**
 * 포트 문자열 해석 — 숫자가 아니거나 범위 밖이면 null(기본값으로 강등).
 * 오타 난 env 하나 때문에 서버가 안 뜨는 것보다, 기본 포트로라도 뜨는 쪽이
 * 현장에서 복구하기 쉽다(로그로 확인 가능).
 */
export function parsePort(raw: string | undefined): number | null {
  if (raw === undefined || raw.trim() === '') return null;
  const n = Number(raw.trim());
  if (!Number.isInteger(n) || n < 1 || n > 65535) return null;
  return n;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  // 상대 경로는 프로세스 cwd 기준으로 절대화한다 — Docker(/app)와 로컬(저장소 루트)
  // 모두 "실행한 곳 아래 server/data"라는 같은 의미가 된다.
  const dataDir = resolve(env.WORKCELL_DATA_DIR?.trim() || DEFAULT_DATA_DIR);
  return {
    port: parsePort(env.WORKCELL_PORT) ?? DEFAULT_PORT,
    dataDir,
    dbPath: resolve(dataDir, 'workcell.db'),
    serverName: env.WORKCELL_SERVER_NAME?.trim() || DEFAULT_SERVER_NAME,
  };
}
