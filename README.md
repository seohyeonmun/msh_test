## Novelist Fine-tuning

- 소설 작가 페르소나를 위한 Qwen3-8B 기반 LoRA/QLoRA 파인튜닝 코드입니다.
---
### 폴더 구성

ai/novelist/
├── train_qlora.py                # 4bit NF4 QLoRA 학습 코드
├── train_lora_bf16.py            # 최종 bf16 LoRA 학습 코드
├── test_lora.py                  # 학습된 LoRA 어댑터 추론 테스트 코드
├── dataset_split.py              # 데이터셋 train/valid/test 분할 코드
├── calculate_dataset_stats.py    # 평균 프롬프트/답변 토큰 길이 계산 코드
├── requirements.txt              # pip 환경 패키지 목록
└── README.md                     # conda 환경 설정 파일

>   ※ train_qlora.py
>    - 4bit NF4 양자화 적용
>    - BitsAndBytesConfig 사용
>    - quantization_config=bnb_config 적용
>    - optim="paged_adamw_8bit"
>
>    ※ train_lora.py
>    - 4bit 양자화 미적용
>    - 모델을 torch_dtype=torch.bfloat16으로 로드
>    - quantization_config 제거
>    - optim="adamw_torch"

---
### 학습 방식

- 초기에는 4-bit NF4 QLoRA 방식으로 학습을 진행했으며, 이후 출력 안정성을 비교하기 위해 bf16 LoRA 방식도 실험했습니다.

- 최종적으로는 Qwen3-8B 모델을 bf16으로 로드하고 LoRA adapter만 학습하는 방식을 사용했습니다.

---
### 주요 설정 (최종 설정)

- Base model: `Qwen/Qwen3-8B`
- Final method: bf16 LoRA
- Epochs: 2
- Learning rate: 5e-5
- Batch size: 1
- Gradient accumulation steps: 8
- Optimizer: `adamw_torch`
---
### 실행 환경

- Python 3.10
- PyTorch 2.5.1+cu121
- Transformers 4.51.3
- TRL 0.17.0
- CUDA 12.1
---
### 설치
1. conda 환경 생성
```bash
conda env create -f environment.yml
conda activate qwen_sh

2. 또는 pip로 직접 설치
- PyTorch는 CUDA 버전에 맞춰 먼저 설치합니다.
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- 이후 필요한 패키지를 설치합니다.
```bash
pip install -r requirements.txt

---
### 주의사항

- 모델 가중치, LoRA adapter, GGUF 파일은 용량 문제로 GitHub에 포함하지 않았습니다.

- 데이터셋은 GitHub에 포함하지 않았으므로, 실행 전 코드 안의 데이터셋 경로를 각자 환경에 맞게 수정해야 합니다.
