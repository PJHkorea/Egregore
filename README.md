
# 🚀 Optax 기반 가속기 최적화 엔진 아키텍처 개편 (v2.0)

본 문서는 `CR_egregore_jax_test.py`에서 `CR_egregore_jax_test_v2.py`로의 마이그레이션을 통해 달성한 **Optax 옵티마이저 엔진 아키텍처 개편**과 관련된 기술적 명세 및 수리 역학적 개선 사항을 다룹니다. LLM 및 MoE 가속화 과정에서 발생하던 XLA 그래프 파편화와 호스트 메모리 누수 문제를 전산학적으로 해결했습니다.

---

## 1. 최적화 엔진: 단일 융합 커널(Fused Kernel) 형성

### ❌ 구 버전 (`CR_egregore_jax_test.py`): 다중 분기 (Multi-Transform) 레일
* **구현 방식**: `optax.multi_transform`을 호출하여 파이썬 딕셔너리 키 경로를 기반으로 `backbone`과 `gate` 노드를 물리 분기했습니다.
* **문제점**: XLA 컴파일러가 장치(Device) 단에서 수많은 명령어 분기 흐름을 생성하여 **그래프 파편화(Graph Fragmentation)** 및 분기 스톨을 유발했습니다.

### ✨ 신 버전 (`CR_egregore_jax_test_v2.py`): 순수 실리콘 MUX 레일
* **구현 방식**: 가중치 감쇠를 배제한 단일 `optax.adam(learning_rate=1.0)` 엔진으로 적률(Momentum)을 깨끗하게 통합 추적합니다.
* **개선 효과**: 외부에서 가속기 ALU가 가장 선호하는 f32 리터럴 마스크와 **아다마르 곱(Hadamard Product)** 대수식만으로 인라인 제어합니다. 이를 통해 분기 스톨이 존재하지 않는 **단일 융합 커널(Fused Kernel)**을 강제 형성합니다.

---

## 2. 가중치 감쇠 수리 역학: 이중 감쇠(Double-Dipping) 박멸

### ❌ 구 버전 (`CR_egregore_jax_test.py`)
* **구현 방식**: 분기된 하위 옵티마이저 내부에서 개별적으로 `optax.adamw`를 호출했습니다.
* **문제점**: 계층별 차등 학습률(LLRD) 마스크와 결합할 때, Adam 내부 모멘텀 계산 식에 가중치 감쇠 계수가 원천 왜곡되어 유입되는 **이중 감쇠(Double-Dipping)** 결함이 발생했습니다.

### ✨ 신 버전 (`CR_egregore_jax_test_v2.py`)
* **구현 방식**: (`PJHkorea/egregore-core-jax/README_OPTIMIZERS.md`)에 증명된 대수적 부호 합치 공식을 코어 세그먼트에 이식했습니다.
* **개선 효과**: 가속기 메모리상에서 모멘텀 왜곡 없이 오리지널 AdamW LLRD 공식 규격을 재현합니다.

$$\text{Update} = (u \times \text{lr}) + (p \times \text{wd} \times \text{wd-activation-gate} \times \text{lr})$$






*   $u$: Adam 모멘텀 업데이트 벡터
*   $\text{lr}$: 계층별 차등 학습률 (LLRD)
*   $p$: 현재 가중치 매개변수 (Parameter)
*   $\text{wd}$: 가중치 감쇠 계수 (Weight Decay)

> 💡 **전산학적 부호 합치 보충 명세**
> `optax.adam` 백엔드가 자체 연산을 거쳐 반환하는 `updates` ($u$) 벡터는 가축치에 바로 더해질 수 있도록 이미 **음수 변위 방향성(Negative Gradient Direction)**이 내장되어 있습니다. 따라서 가중치를 물리적으로 줄여야 하는 Weight Decay 성분의 음수 스케일링 방향과 무결하게 동기화하기 위해 대수학적으로 빼기($-$)가 아닌 **더하기($+$) 기호로 결합**하는 것이 수치해석적으로 완전히 올바른 구현입니다.

---

## 3. 호스트 메모리 인프라: 중복 트레이싱 및 타입 크래시 제로화

### ❌ 구 버전 (`CR_egregore_jax_test.py`)
* **구현 방식**: LLRD 스케일러 자체가 내장되어 있지 않아 복잡한 하이퍼파라미터 분기 로직에 의존했습니다.
* **문제점**: 거대 모델 컴파일 시 호스트 CPU의 정적 트레이싱 오버헤드가 급증했습니다.

### ✨ 신 버전 (`CR_egregore_jax_test_v2.py`)
* **구현 방식**: `tree_flatten_with_path` 호출을 스케줄러 루프 내에서 정확히 **단 1회**로 한계 수축했습니다.
* **기믹 도입**: `mapped_scalars`와 원본 리프 가중치 v를 `zip`으로 묶어 텐서를 사후 확장하는 **`[LEAF-LEVEL TENSOR RECONSTRUCTION]`** 매커니즘을 정착시켰습니다.
* **개선 효과**: 거대 LLM/MoE 모델 컴파일 시 발생하던 호스트 CPU의 메모리 누수 및 **Host OOM(Out-Of-Memory) 크래시를 전산학적으로 해결**했습니다.

---

## 📊 요약 및 지표 비교

| 평가 항목 | 구 버전 (`CR_egregore_jax_test.py`) | 신 버전 (`CR_egregore_jax_test_v2.py`) | 비고 |
| :--- | :--- | :--- | :--- |
| **옵티마이저 구조** | `optax.multi_transform` (다중 분기) | 단일 `optax.adam` + 인라인 마스킹 | 그래프 최적화 |
| **XLA 커널 형태** | 파편화된 명령어 분기 흐름 | **단일 융합 커널 (Fused)** | 분기 스톨 제로화 |
| **수리적 무결성** | Adam 내부 모멘텀 왜곡 발생 | **AdamW LLRD 규격 100% 재현** | 대수적 부호 합치 |
| **호스트 컴파일** | 중복 트레이싱 및 Host OOM 위험 | **1회 수축 및 리프 레벨 재구성** | CPU 메모리 누수 박멸 |
