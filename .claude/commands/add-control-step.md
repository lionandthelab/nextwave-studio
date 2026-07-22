새 제어 step 종류를 추가한다. CLAUDE.md §6의 워크플로를 엄격히 따른다.

추가할 step: $ARGUMENTS

절차:
1. **설계 확인**: docs/DATA_MODEL.md §6과 docs/SIMULATION.md §3을 읽고, 새 step이
   기존 `ControlStep` 유니온·player 해석 모델과 일관되는지 확인한다.
2. **스키마 우선**: `src/schema`의 `ControlStep` 유니온에 새 variant를 추가하고
   런타임 검증(필수 필드, robot/joint 참조 무결성)을 작성한다. tsc 통과 확인.
3. **핸들러**: `src/core/control/steps.ts`에 해당 step 핸들러를 구현한다.
   - 보간형이면 시작값 스냅샷 + easing + `elapsedSec` 진행 → 완료 시 advance.
   - 즉시형이면 적용 후 즉시 advance.
   - 배리어형이면 조건 충족/타임아웃까지 대기.
4. **라우팅**: `player.ts`의 switch에 새 kind를 연결한다.
5. **로봇 종류 대응**: kinematicPosition(setpoint 직접) / dynamic(PD 목표) 양쪽에서
   의미가 성립하는지 확인한다. MVP는 kinematic 우선.
6. **샘플 갱신**: 샘플 ControlSequence(JSON)에 사용 예시를 추가한다.
7. **검증**: 단위 테스트(순수 진행 로직) + 실제 재생으로 동작 확인.
8. **기록**: EXPERIMENTS.md에 설계 결정을 남긴다.

불변식(CLAUDE.md §2.6): 특정 로봇 동작을 엔진 코드에 하드코딩하지 말 것. step은
선언적 데이터로 표현되고 player가 해석해야 한다.

완료 전 `/verify-gate`를 실행한다.
