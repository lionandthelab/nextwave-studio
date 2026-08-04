# 3D 임포트 시험용 모델 — 출처 · 라이선스

이 디렉터리의 모델은 **3D 파일 임포트 파이프라인**(`src/render/mesh-import.ts`,
`src/ui/library/import-dialog.ts`)을 실제로 시험하기 위한 샘플이다.
`.glb` / `.stl` / `.obj` 세 포맷이 각각 다른 three 로더(GLTFLoader / STLLoader / OBJLoader)를
타므로, **포맷별로 최소 1개씩** 확보했다.

---

## 1. 라이선스 요약

**모든 파일은 원본이 CC0 1.0 Universal (public domain dedication)이다.**
CC0는 저작권·인접권을 전 세계적으로 포기하는 선언이므로 **출처 표기 의무가 없고, 상업적
이용·수정·재배포가 자유롭다.** 아래 표기는 의무가 아니라 추적 가능성을 위한 기록이다.

원본 출처는 전부 Khronos Group의 **glTF-Sample-Assets** 저장소다.

- 저장소: https://github.com/KhronosGroup/glTF-Sample-Assets
- 확인 시점의 `main` 커밋: `2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf` (2026-04-27T15:55:06Z)
- 내려받기 시각 기준: 2026-08-04

### 라이선스를 확인한 방법 (추측 아님)

이 저장소는 **모델 디렉터리마다 `LICENSE.md`를 둔다.** 각 모델의 `LICENSE.md`를 직접
내려받아 본문을 읽어 확인했다. 파일 형식은 아래와 같고, **첫 번째 항목이 모델 파일 자체**,
두 번째 항목이 메타문서(`LICENSE.md`, `metadata.json`)에 적용된다.

```
* All files directly associated with the model including all text, image and binary files:

  * [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
    [SPDX license identifier: "CC0-1.0"]

* This file and all other metadocumentation files including "metadata.json":

  * [Creative Commons Attribution 4.0 International] [SPDX license identifier: "CC-BY-4.0"]
```

> **주의 — 이 저장소는 모델마다 라이선스가 다르다.** 전체 148개 모델의 `LICENSE.md`를
> 모두 조회한 결과 **모델 파일 자체가 CC0-1.0인 것은 54개뿐**이고, 나머지는 CC-BY-4.0,
> CC-BY-NC-SA, Poser EULA, CRYENGINE 계약, Adobe Stock 라이선스 등이다.
> 예를 들어 흔히 쓰이는 `Box`, `Duck`, `CesiumMilkTruck`, `DamagedHelmet`은 **CC0가 아니다.**
> 여기에는 CC0로 확인된 것만 넣었다. 새 모델을 추가할 때도 반드시 해당 모델의
> `LICENSE.md`를 직접 읽고 확인할 것.
>
> 두 번째 항목의 CC-BY-4.0은 **메타문서에만** 적용되며, 우리는 메타문서를 가져오지
> 않았으므로 이 디렉터리에 CC-BY 의무가 붙는 파일은 없다.

---

## 2. 파일 목록

| 파일 | 포맷 | 크기 | 삼각형 | 실제 치수 (X×Y×Z, m) | 원본 모델 | 라이선스 |
|---|---|---:|---:|---|---|---|
| `avocado.glb` | glTF 바이너리 | 14.8 KB | 682 | 0.043 × 0.063 × 0.028 | Avocado | CC0-1.0 |
| `water-bottle.glb` | glTF 바이너리 | 89.2 KB | 4,510 | 0.109 × 0.260 × 0.109 | WaterBottle | CC0-1.0 |
| `teacup.stl` | 바이너리 STL | 832.5 KB | 16,648 | 0.106 × 0.067 × 0.132 | DiffuseTransmissionTeacup | CC0-1.0 |
| `boombox.stl` | 바이너리 STL | 301.9 KB | 6,036 | 0.020 × 0.020 × 0.020 | BoomBox | CC0-1.0 |
| `barramundi-fish.obj` | Wavefront OBJ | 398.2 KB | 3,864 | 0.143 × 0.288 × 0.643 | BarramundiFish | CC0-1.0 |

합계 **1.64 MB**.

### 원본 URL (모델 / 라이선스)

| 파일 | 원본 모델 URL | LICENSE.md URL |
|---|---|---|
| `avocado.glb` | [Models/Avocado/glTF-Binary/Avocado.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Avocado/glTF-Binary/Avocado.glb) | [Models/Avocado/LICENSE.md](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Avocado/LICENSE.md) |
| `water-bottle.glb` | [Models/WaterBottle/glTF-Binary/WaterBottle.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/WaterBottle/glTF-Binary/WaterBottle.glb) | [Models/WaterBottle/LICENSE.md](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/WaterBottle/LICENSE.md) |
| `teacup.stl` | [Models/DiffuseTransmissionTeacup/glTF-Binary/DiffuseTransmissionTeacup.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/DiffuseTransmissionTeacup/glTF-Binary/DiffuseTransmissionTeacup.glb) | [Models/DiffuseTransmissionTeacup/LICENSE.md](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/DiffuseTransmissionTeacup/LICENSE.md) |
| `boombox.stl` | [Models/BoomBox/glTF-Binary/BoomBox.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/BoomBox/glTF-Binary/BoomBox.glb) | [Models/BoomBox/LICENSE.md](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/BoomBox/LICENSE.md) |
| `barramundi-fish.obj` | [Models/BarramundiFish/glTF-Binary/BarramundiFish.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/BarramundiFish/glTF-Binary/BarramundiFish.glb) | [Models/BarramundiFish/LICENSE.md](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/BarramundiFish/LICENSE.md) |

---

## 3. 원본에 가한 수정 (전부 CC0가 허용하는 파생)

원본 `.glb`는 2K 텍스처 때문에 **개당 4.8–12.5 MB**라 저장소에 넣기에 너무 크다.
저장소에 들어가는 바이너리이므로 **지오메트리만 남기고 재작성**했다.

**공통 — 텍스처 제거 (모든 파일):**

- glTF JSON에서 `images` / `textures` / `samplers`를 삭제하고, 머티리얼의 텍스처 참조
  (`baseColorTexture`, `metallicRoughnessTexture`, `normalTexture`, `occlusionTexture`,
  `emissiveTexture`)와 머티리얼 확장(`KHR_materials_*`)을 제거했다.
- 텍스처가 없으므로 정점 속성 `TEXCOORD_n` / `TANGENT`도 제거했다
  (`POSITION` / `NORMAL` / `COLOR_0`만 유지).
- 흰색 `baseColorFactor`는 단색으로는 형체가 안 보이므로 모델마다 대략적인 고유색으로
  대체했다 (예: 아보카도 짙은 녹색, 물병 청색).
- 사용되지 않게 된 `bufferView`를 제거하고 BIN 청크를 재구성했다.

**포맷 변환:**

- `teacup.stl` — 원본 glb는 메시가 2개(`tea_cup` 16,648 tris + `tea_saucer` 11,536 tris,
  합계 28,184)다.
  **찻잔(`tea_cup`)만** 추출해 three `STLExporter`(binary)로 내보냈다.
  받침을 포함하면 파일이 1 MB를 넘어 제외했다.
- `boombox.stl` — three `STLExporter`(binary).
- `barramundi-fish.obj` — three `OBJExporter`.
  (참고: 찻잔을 OBJ로 내보내면 인덱스가 풀려 7 MB가 되므로 OBJ로는 쓰지 않았다.)

**좌표/스케일은 건드리지 않았다.** glTF는 규격상 Y-up · 미터이므로 파생된 STL/OBJ도
그대로 Y-up · 미터다.

---

## 4. 임포트할 때의 스케일 (arm6 작업 반경 ≈ 0.5 m 기준)

임포트 다이얼로그(`src/ui/library/import-dialog.ts`)에서 **`up 축 = Y`** 를 고르면 된다
(위 3절대로 전부 Y-up이다. Z-up을 고르면 -90° 눕는다).

| 파일 | 원본 치수 | 권장 스케일 | 스케일 후 | 비고 |
|---|---|---:|---|---|
| `avocado.glb` | 6.3 cm 높이 | **1.0** | 그대로 | 그리퍼로 집기 좋은 크기. 기본 시험 대상으로 가장 적합 |
| `water-bottle.glb` | 26.0 cm 높이 | **1.0** | 그대로 | 실물 크기. 세워두면 넘어뜨리기 시험에 좋다 |
| `teacup.stl` | 13.2 cm 폭 | **1.0** | 그대로 | 실물 크기 찻잔 |
| `boombox.stl` | **2.0 cm** | **10.0** | 20 cm | 원본이 2 cm로 아주 작게 제작돼 있다(실물 붐박스는 40–50 cm이므로 실물 크기로는 20–25배). 스케일 필드를 실제로 시험하기 가장 좋은 사례 |
| `barramundi-fish.obj` | **64.3 cm 길이** | **0.5 정도** | 32 cm | 원본이 작업 반경(0.5 m)보다 길다. 1.0으로 넣으면 팔보다 크다 |

---

## 5. 검증 기록 (2026-08-04)

저장소의 실제 임포트 코드(`src/render/mesh-import.ts`의 `parseModelFile` →
`prepareForScene`)를 Node(vitest, three 0.169)에서 **5개 파일 전부에 대해 실행해**
파싱·삼각형 수·바운딩 박스·convex hull 정점 수·피벗 재정렬을 확인했다. 전부 통과.

| 파일 | 감지된 형식 | 삼각형 | hull 정점 | half extents (m) | 피벗 오프셋 (m) |
|---|---|---:|---:|---|---|
| `avocado.glb` | `glTF (.glb)` | 682 | 363 | 0.0213 / 0.0314 / 0.0138 | 0 / 0.0314 / 0 |
| `water-bottle.glb` | `glTF (.glb)` | 4,510 | 1,508 | 0.0545 / 0.1302 / 0.0545 | 0 / 0.1302 / 0 |
| `teacup.stl` | `STL (.stl)` | 16,648 | 1,959 | 0.0532 / 0.0336 / 0.0661 | 0 / 0.0336 / 0 |
| `boombox.stl` | `STL (.stl)` | 6,036 | 1,650 | 0.0099 / 0.0098 / 0.0101 | 0 / 0.0098 / 0 |
| `barramundi-fish.obj` | `OBJ (.obj)` | 3,864 | 1,383 | 0.0715 / 0.1438 / 0.3216 | 0 / 0.1438 / 0 |

- hull 정점 수는 전부 `MAX_HULL_POINTS`(2,048) 이하 — 데시메이션이 정상 동작한다.
- 피벗 오프셋이 `[0, halfY, 0]`이므로 엔티티 `position.y = 0`에 놓으면 바닥에 정확히 앉는다.
- 변환된 STL/OBJ의 삼각형 수와 바운딩 박스는 원본 glb와 일치한다(찻잔은 받침 제외분만큼 차이).

### 무결성 (SHA-256)

```
a20f9ec5bd69905b1c84d36cc5eb687e640a09cf00a2fc091a0c7a64d1b02f2d  avocado.glb
27b166022b4ba74444e3497ff16af3f601bcd69cdff6ed5386307fd177eedb35  barramundi-fish.obj
22aaec7a71112c4ae9a24193a7c3d9142092c460303792e4ea2ab8f63f35525b  boombox.stl
75246b3f75a62f346f4f3a560a7a133ec1b8b5b68eaba6a78cd3e2f579d2050f  teacup.stl
b5f310fc3352c875c9a585e54e136ed9418cbecf3db8395484ddfa3745223627  water-bottle.glb
```

원본(내려받은 그대로, 텍스처 포함)의 SHA-256:

```
ccc9c3ce56423720b09399c2351537207cd5a65f859f9e6e2f30922762f3abd4  Avocado.glb
ecc3bafb6b00f2c8b810863c388e3768a7b7ea0d0335e8cb8c574c266e571f4a  BarramundiFish.glb
f8b918445ebdd006768232205a62f5182d2208ca57f84c6ccc084943c0bc8f15  BoomBox.glb
d4f567186fea262819aac6664cb9d7cef7f4600406038046771a69e10363935b  DiffuseTransmissionTeacup.glb
b337e526fd6a162013c2984aeec163f5fbb4f717252724dfc3f3458bd51df94b  WaterBottle.glb
```

---

## 6. 알려진 한계

- **임포트 에셋은 세션 한정이다.** `asset://` 참조는 현재 세션의 `MeshAssetStore`에서만
  해석된다. 씬을 저장해도 메시 데이터가 파일에 내장되지 않으므로 다른 세션에서 열면
  해당 엔티티가 복원되지 않는다 (`mesh-import.ts`의 `ASSET_SAVE_WARNING_KO`).
- **`.gltf`(외부 `.bin`/텍스처 참조)는 지원하지 않는다.** 단일 파일 전용이므로
  여기에는 `.glb`만 두었다.
- **STL/OBJ는 단일 파일에 재질을 담지 못한다.** STL은 형식 자체가 재질이 없고, OBJ는
  재질을 별도 `.mtl`에 두는데 이 앱은 단일 파일 임포트다. 둘 다 임포트 파이프라인이
  같은 기본 재질(`IMPORTED_MESH_COLOR` MeshStandardMaterial)을 씌운다
  (`src/render/mesh-import.ts`). 예전에는 OBJ만 OBJLoader 기본값(흰색 MeshPhongMaterial)이
  남아 씬의 PBR 조명 아래 혼자 납작하게 떴다 — 지금은 STL과 한 모양으로 모았다.
