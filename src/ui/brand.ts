// ui/brand.ts — 제품 정체성 단일 진실 (UX_AUDIT C-12)
//
// 구 상태: <title>이 패키지 slug("robot-sim-web"), 파비콘 없음, 실사용 <h1> 0개,
// 브랜드 마크는 8×8px 사각형 + 13px 텍스트. 스크린샷·데모·공유 링크 어디에도 제품이
// 없었고, 프로 사용자가 첫 0.5초에 "만들다 만 것"으로 판정하는 근거가 됐다.
//
// 이름을 바꾸려면 이 파일의 상수만 고치면 된다 — 다른 곳에 제품명 리터럴을 흩지 말 것.
//
// ── 네이밍 근거 ─────────────────────────────────────────────────────
// **workcell**은 "로봇 + 지그 + 부품이 한 셀에 놓인 작업 단위"를 뜻하는 로보틱스 표준
// 용어로, SceneSpec이 기술하는 대상과 1:1로 일치한다. 저장 파일 확장자(.workcell.json)와
// UI 명사("워크셀 열기")가 자연스럽게 따라 나와, "이 파일이 전부다"라는 약속을 이름이
// 강제한다. 상위 브랜드는 워크스페이스 경로(lionandthelab/nextwave-studio)의 NextWave가
// 흡수한다.

/** 제품명 — 탭 제목·워드마크·문서 타이틀의 단일 진실 */
export const PRODUCT_NAME = 'Workcell';

/** 상위 브랜드 (푸터·정보 표시용) */
export const BRAND_NAME = 'NextWave Studio';

/** 한 줄 설명 — <meta name="description">과 빈 상태 헤드라인에 쓰인다 */
export const PRODUCT_TAGLINE = '브라우저에서 완결되는 로봇 시뮬레이션 스튜디오';

/** 저장 파일 확장자 — {scene, sequence, assets} 봉투 문서 */
export const DOCUMENT_EXTENSION = '.workcell.json';

/** localStorage / IndexedDB 키 접두 */
export const STORAGE_PREFIX = 'workcell';

/**
 * 브라우저 탭 제목을 갱신한다.
 *
 * 형식: `{● }{문서명} — {제품명}` (● = 미저장 변경 있음)
 * 문서명이 없으면 `{제품명} — {한 줄 설명}`.
 */
export function setDocumentTitle(documentName: string | null, dirty = false): void {
  if (documentName === null || documentName === '') {
    document.title = `${PRODUCT_NAME} — ${PRODUCT_TAGLINE}`;
    return;
  }
  document.title = `${dirty ? '● ' : ''}${documentName} — ${PRODUCT_NAME}`;
}
