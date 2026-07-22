현재 작업을 "완료"로 보고하기 전에 검증 게이트를 통과했는지 점검한다.

다음을 순서대로 확인하고, 각 항목의 상태를 정직하게 보고한다(통과/미통과/해당없음):

**공통 Definition of Done (CLAUDE.md §8)**
1. `npm run typecheck` (tsc --noEmit) 통과
2. `npm run lint` 경고 0
3. 관련 순수 로직 단위 테스트(vitest) 통과
4. 샘플 씬 로드 + 지정 프레임레이트에서 물리 안정성 확인
5. 의도한 충돌 쌍이 EventQueue로 실제 감지되어 로그에 남는지 확인
6. 불변식(CLAUDE.md §2) 위반 없음
7. 계층 의존 방향(CLAUDE.md §3, ui → core → {render, schema}) 준수
8. 결정/변경 사항을 EXPERIMENTS.md에 기록

**현재 Phase의 게이트 (docs/ROADMAP.md)**
- 지금 작업이 속한 Phase의 게이트 항목을 확인하고 통과 여부를 명시한다.

**Conservative Self-Critique (AGENTS.md §5, 거부권)**
- 게이트를 실제로 통과하지 않았다면 "완료"라고 주장하지 말 것.
- 알려진 한계·미해결 문제를 감추지 말고 명시할 것.
- 성능/정확도 주장에는 측정 근거를 붙일 것.

미통과 항목이 있으면 완료 보고 대신, 남은 작업을 구체적으로 나열한다.

대상: $ARGUMENTS
