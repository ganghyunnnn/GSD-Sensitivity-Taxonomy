"""
GSD-Sensitivity Taxonomy Definitions
RS 태스크를 D / M1 / M2 세 유형으로 분류하는 기준 정의

핵심 기준:
    분류는 '출력 연산'에 의존하며, 쿼리의 주제·필터와 무관하다.
    비교·최상급은 그 기반이 되는 M1/M2에 흡수된다.
"""

from enum import Enum


class TaskType(str, Enum):
    D  = "D"   # Description: 시각적·의미적 해석만으로 답 가능
    M1 = "M1"  # Spatial Metric: GSD-dependent 공간 연산 필요
    M2 = "M2"  # Cardinality: 정밀 카운팅 필요


TAXONOMY_DEFINITIONS = {
    TaskType.D: {
        "name": "Description",
        "description": (
            "시각적·의미적 해석만으로 답 가능. GSD가 바뀌어도 정답이 변하지 않는다. "
            "포함 범위: 씬 서술, 객체 유무/범주, bbox, 색상/형태, 방향/방위, "
            "시각적 예/아니오, 정성적 비교, 정성적 공간 관계."
        ),
        "examples": [
            "What type of land use is this area?",
            "Is there an airport in this image?",
            "What is the heading of the aircraft?",
            "Draw a bounding box around the flooded houses.",
            "Which side has more damage? (no GSD)",
        ],
    },
    TaskType.M1: {
        "name": "Spatial Metric",
        "description": (
            "어떤 단계에서든 GSD-dependent 공간 연산이 필요한 태스크. "
            "(a) 출력이 실측 단위 측정값(거리·면적·길이)이거나, "
            "(b) 실측 단위 공간 조건(50m 이내 등)을 GSD로 평가해야 한다. "
            "비교·최상급도 GSD 측정에 기반하면 M1에 흡수."
        ),
        "examples": [
            "What is the distance between A and B? GSD=0.3m/px",
            "Is there a house within 50m of the car? GSD=x",
            "Which parking lot is larger? GSD=x",
            "What is the area of the damaged region in sq meters? GSD=x",
        ],
        "trigger_keywords": [
            "distance", "meter", "metre", "km", "feet", "foot", "area", "length",
            "width", "height", "gsd", "m/px", "meters away", "within \\d+m",
            "how far", "how long", "how wide", "size of", "square meter",
        ],
    },
    TaskType.M2: {
        "name": "Cardinality",
        "description": (
            "이산 객체를 정밀하게 세야 하는 태스크. "
            "비율·백분율·카운트 기반 산술도 M2에 포함. "
            "카운트 기반 비교·최상급도 M2에 흡수."
        ),
        "examples": [
            "How many cars are in the parking lot?",
            "Count the number of buildings attached to the cul de sac.",
            "Which side has more cars?",
            "What percentage of buildings are destroyed?",
            "What is the ratio of damaged to undamaged buildings?",
        ],
        "trigger_keywords": [
            "how many", "count", "number of", "total number", "quantity",
            "how much", "enumerate", "ratio", "percentage", "proportion",
        ],
    },
}


# ThinkGeo 도구 → Taxonomy 유형 매핑 (annotation 보조용)
TOOL_TO_TAXONOMY_HINT = {
    "Calculator": [TaskType.M1, TaskType.M2],
    "Solver": [TaskType.M1],
    "CountGivenObject": [TaskType.M2],
    "TextToBbox": [TaskType.D],
    "SegmentObjectPixels": [TaskType.M1],
    "ChangeDetection": [TaskType.D],
    "RegionAttributeDescription": [TaskType.D],
    "ObjectDetection": [TaskType.M2, TaskType.D],
    "ImageDescription": [TaskType.D],
    "GoogleSearch": [TaskType.D],
    "OCR": [TaskType.D],
    "DrawBox": [TaskType.D],
    "AddText": [TaskType.D],
    "Plot": [TaskType.M2],
}
