새 샘플 씬을 **데이터만으로** 추가한다(코드 변경 없이). CLAUDE.md §2.5 불변식.

씬 설명: $ARGUMENTS

절차:
1. docs/DATA_MODEL.md의 `SceneSpec` 스키마(§5)와 예시(§9)를 따른다.
2. `src/assets/scenes/`에 새 `*.scene.json`을 작성한다.
   - 좌표계: Y-up, 미터, 라디안. 회전은 쿼터니언 [x,y,z,w].
   - gravity, timestepHz(예: 240), camera, environment 설정.
   - 각 엔티티: id(유일), type, transform, visual, physics.
   - collider 그룹/`collidesWith`는 CLAUDE.md §5 규약을 따른다.
   - 충돌을 감지할 collider에는 `emitEvents: true` + 상대 그룹 포함.
   - 동적 바디에 `trimesh` 금지(프리미티브/convexHull 사용).
3. 필요하면 대응 `*.sequence.json`(ControlSequence)도 함께 작성한다.
   - 시퀀스의 robot/joint 참조가 씬에 실제 존재하는지 확인.
4. 스키마 검증을 통과하는지 확인한다(참조 무결성·그룹 유효성·필수 필드).
5. 로드해서 렌더·재생·충돌 감지가 의도대로 동작하는지 확인한다.

코드(엔진/로더)를 수정해야만 이 씬이 로드된다면, 그것은 스키마/로더의 표현력
부족 신호다. 하드코딩 대신 스키마 확장을 검토하고 EXPERIMENTS.md에 남긴다.
