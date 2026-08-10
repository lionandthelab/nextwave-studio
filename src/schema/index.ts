// schema/index.ts — 스키마 계층 배럴.
// 타입(types.ts) + 런타임 검증 API(validate.ts) + Flow Graph 뷰 모델(flow-graph.ts)을
// 한 지점에서 재수출한다. 다른 계층은 `src/schema`에서 import하는 것을 권장한다.

export * from './types';
export * from './validate';
export * from './flow-graph';
// Phase 12+ — 협업 개체(공정/작업/블록/장비/실행기록) 계약 + 재사용 블록 캡처/전개
export * from './entities';
export * from './blocks';
