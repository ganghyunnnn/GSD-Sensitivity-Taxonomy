# RS Taxonomy 재주석 가이드라인
## Inter-Annotator Agreement (IAA) 작업 안내

**작업 개요**: 원격탐사 VQA 질문 88개를 읽고, 아래 분류 기준에 따라 타입을 지정합니다.
**소요 시간**: 약 1–1.5시간 예상
**제출 파일**: `iaa_sample.csv` (type_annotator 컬럼 작성 완료본)

---

## 분류 체계 (3가지 타입)

| 타입 | 이름 | 한 줄 정의 |
|------|------|-----------|
| **D** | Descriptive (설명형) | 이미지를 보고 시각적으로 판단하면 답할 수 있는 질문 |
| **M1** | Spatial Metric (공간 측정형) | GSD(지상 해상도)를 이용한 실제 거리·면적 계산이 필요한 질문 |
| **M2** | Cardinality (계수형) | 객체를 정확히 세는 것이 필요한 질문 |

---

## 핵심 판단 기준

> **답변에 무엇이 필요한가?** 로만 판단합니다. 질문의 주어(홍수, 자동차 등)는 타입에 영향을 주지 않습니다.

### D — 시각적 판단만으로 답할 수 있는가?

다음 중 하나면 D:
- 객체의 존재 여부 (있다/없다, yes/no)
- 종류·카테고리 (어떤 건물인가, 토지 이용 유형 등)
- 방향·위치 (동쪽, 북서쪽 등)
- 시각적 비교 (더 커 보이는가? — GSD 없이)
- **bounding box 그리기** (항상 D)
- 장면 묘사·설명

```
"Is there a swimming pool?"              → D  (존재 여부)
"What is the heading of the aircraft?"   → D  (방향)
"Draw a box around the flooded houses."  → D  (bbox 출력)
"Which area looks larger?" (GSD 없음)    → D  (시각적 비교)
```

---

### M1 — GSD를 이용해 실제 거리·면적을 계산해야 하는가?

다음 중 하나면 M1:
- 질문에 **GSD 값이 명시**되어 있고, 그 값을 실제로 사용해야 함
- 실제 단위(m, km, m²)로 된 거리·면적·길이를 구해야 함
- "~m 이내" 같은 **실제 단위 기준의 공간 조건**이 포함됨

```
"What is the distance between A and B? GSD = 0.3 m/px"  → M1
"How far apart are the two buildings? GSD = 0.5 m/px"   → M1
"How many trees within 50m of the pool? GSD = 0.3"      → M1+M2 (아래 복합 참조)
"Which building is larger? GSD = 0.3 m/px"              → M1
```

> ⚠️ **GSD가 언급되어도 실제로 안 쓰면 M1이 아닙니다.**
> `"Describe the scene. GSD = 0.3 m/px"` → D (GSD를 쓰지 않음)

---

### M2 — 객체를 정확히 세어야 하는가?

다음 중 하나면 M2:
- "몇 개", "how many", "count"
- 개수 기반 비율·퍼센트 ("전체 건물의 몇 %가 손상되었나")
- 개수 기반 비교·최고급 ("어느 쪽에 차가 더 많은가", "주차장이 가장 많은 곳")

```
"How many cars are in the parking lot?"          → M2
"Which side has more buildings?"                  → M2
"What percentage of houses are flooded?"          → M2
"Count the vehicles near the intersection."       → M2
```

---

## 복합 타입 (두 가지 이상 동시 해당)

두 가지 독립적인 출력이 모두 필요할 때만 복합으로 표기합니다.

### M1+M2
계수(M2)와 거리 기준(M1)이 **둘 다** 필수인 경우:

```
"How many buildings are within 50m of the river? GSD = 0.3"
→ M1+M2  (50m 경계 설정에 GSD 필요 + 건물 개수 세기 필요)
```

### D+M2
시각적 결과물(D)과 개수(M2)가 **둘 다** 명시적으로 요구된 경우:

```
"Detect total vehicles (draw bbox) and how many are sedans?"
→ D+M2  (bbox 그리기 + 개수 세기)

"Detect total vehicles and how many are sedans?"
→ M2만  (detect = 개수 세기의 전처리, bbox 요구 없음)
```

> 핵심: **"detect/identify"만 있으면 D가 아닙니다.** bbox나 목록 출력이 명시되어야 D가 포함됩니다.

### D+M1
시각적 결과물(D)과 GSD 계산(M1)이 **둘 다** 명시된 경우 (매우 드문 경우):

```
"Describe the flood damage and calculate the total flooded area. GSD = 0.3"
→ D+M1
```

---

## 작성 방법

### CSV 열기
`annotation/iaa_sample.csv` 파일을 엑셀 또는 Google Sheets로 열고,
`type_annotator` 컬럼에 타입을 입력합니다.

### 입력 형식

| 타입 | 입력값 |
|------|--------|
| Descriptive만 | `D` |
| Spatial Metric만 | `M1` |
| Cardinality만 | `M2` |
| M1과 M2 동시 | `M1+M2` |
| D와 M2 동시 | `D+M2` |
| D와 M1 동시 | `D+M1` |
| 셋 다 | `D+M1+M2` |

대소문자 구분 없습니다 (`m1+m2`도 인식됨).



---

## 빠른 판단 순서도

```
질문을 읽는다
      │
      ▼
GSD 값이 있고, 그 값으로 실제 거리/면적을 계산해야 하나?
  YES → M1 포함
  NO  → ↓
      │
      ▼
객체를 정확히 세어야 하나? (how many / count / 비율 / 개수 비교)
  YES → M2 포함
  NO  → ↓
      │
      ▼
시각적 판단만으로 답할 수 있나?
  YES → D
```

M1과 M2가 **둘 다** 해당하면 → `M1+M2`
bounding box 출력이 **추가로** 명시되면 → D도 포함

---

## 예시 30선 (빠른 감각 익히기)

| 질문 (요약) | 정답 | 이유 |
|------------|------|------|
| Is there a swimming pool? | D | 존재 여부 시각 판단 |
| How many cars are in lot? | M2 | 개수 세기 |
| Distance between A and B? GSD=0.3 | M1 | GSD 거리 계산 |
| Draw box around flooded houses | D | bbox 출력 |
| Which side has more buildings? | M2 | 개수 비교 |
| What land use type is shown? | D | 카테고리 판단 |
| Area of flooded region? GSD=0.5 | M1 | GSD 면적 계산 |
| How many trees within 50m? GSD=0.3 | M1+M2 | 50m 경계(M1) + 개수(M2) |
| What is heading of aircraft? | D | 방향 시각 판단 |
| Count vehicles going northeast | M2 | 방향은 필터, 개수가 목적 |
| Total width of road? (no GSD) | D | GSD 없이 시각 추정 |
| Which quadrant has most cars? | M2 | 개수 기반 최고급 |
| Is the building flooded? | D | 시각적 yes/no |
| How many days to clear 3 sites/day? | M2 | 개수에서 파생된 계산 |
| Length of runway? GSD=0.015 | M1 | GSD 길이 계산 |
| Detect damaged buildings and count | M2 | detect=전처리, 출력은 개수 |
| Detect (draw box) and count | D+M2 | bbox+개수 둘 다 명시 |
| Which building is largest? GSD=x | M1 | GSD 측정 기반 최고급 |
| Ratio of damaged to intact buildings? | M2 | 개수 비율 |
| Are all no-damage buildings in north? | D | 시각적 위치 판단 |
| Describe scene. GSD=0.3 mentioned | D | GSD 실제 미사용 |
| Estimate area ratio A vs B. GSD=x | M1 | GSD 면적 비교 |
| Count cars + describe flood damage | D+M2 | 개수(M2) + 묘사(D) 독립 요구 |
| How far apart? GSD=0.5 m/px | M1 | GSD 거리 |
| Which side has more flooded houses? | M2 | 개수 비교 (홍수는 필터) |
| Is there a house within 50m? GSD=x | M1 | GSD 공간 조건 |
| Draw box on cars going northeast | D | bbox 출력 (방향은 필터) |
| What percentage of buildings destroyed? | M2 | 개수 파생 퍼센트 |
| Longest ship length? GSD=0.26 | M1 | GSD 측정 기반 최고급 |
| How many rescue teams if 3 sites/day? | M2 | 개수 파생 계산 |

---

## 문의

작업 중 판단이 어려운 사례는 `notes` 컬럼에 기록 후 제출해주시면,
최종 확인 후 처리하겠습니다.
