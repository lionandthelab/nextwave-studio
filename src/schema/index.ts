// schema/index.ts — 스키마 계층 배럴.
// 타입(types.ts) + 런타임 검증 API(validate.ts)를 한 지점에서 재수출한다.
// 다른 계층은 `src/schema`에서 import하는 것을 권장한다.

export * from './types';
export * from './validate';
