# [시스템소프트웨어 5분반 15주차] 리눅스 실습 5

**본 리포지터리를 fork하여 코드를 작성한 뒤, 과제란에 본인이 fork한 리포지터리의 링크를 제출한다.**
또는, 리포지터리를 내려받아 코드를 작성한 후 전체를 `zip` 파일로 제출해도 무방하다.

---

## 1. 실습 개요

머신러닝(Machine Learning)에는 다양한 기법이 존재하지만, 다층 퍼셉트론을 포함한 **인공신경망(Neural Network)** 기반 회귀/분류 모델은 일반적으로 다음과 같은 계산 구조를 가진다.

입력 벡터 $x \in \mathbb{R}^d$ 와 정답 레이블 $y \in \mathbb{R}$에 대해, 하나의 은닉층을 갖는 간단한 신경망은 다음과 같이 표현된다.

$$
h = \sigma(W_1 x + b_1), \quad
\hat{y} = w_2^\top h + b_2
$$

여기서

* $W_1$: 입력층에서 은닉층으로 가는 가중치 행렬
* $b_1$: 은닉층 편향
* $w_2$: 은닉층에서 출력층으로 가는 가중치 벡터
* $b_2$: 출력층 편향
* $\sigma(\cdot)$: ReLU 등의 비선형 활성화 함수

손실 함수로 평균제곱오차(Mean Squared Error, MSE)를 사용하면,

$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

와 같이 정의할 수 있으며, 이에 대한 기울기를 이용하여 확률적 경사 하강법(SGD)으로 파라미터를 갱신한다.

실제 머신러닝 시스템에서는 이러한 수치 연산을 보통 다음과 같은 단계 흐름으로 구현한다.

1. **전처리(Preprocessing)**: CSV 파일에서 샘플을 읽고, feature를 정규화한다.
2. **특징 변환(Feature Engineering)**: 비선형 변환(예: 제곱, 조합 등)을 수행한다.
3. **순전파(Forward Pass)**: 입력으로부터 예측값 $\hat{y}$를 계산한다.
4. **역전파(Backward Pass) 및 학습(Training)**: 손실 $L$에 대한 기울기를 계산하고 파라미터를 갱신한다.
5. **로깅/모니터링(Logging)**: 손실, 예측값 등을 집계하여 요약 통계를 출력한다.

본 실습에서는 다음과 같은 전제가 주어진다.

* **정규화, feature 증강, 신경망 forward/backward, SGD 갱신 등 수치 연산은 이미 구현되어 있다.**
* 학생은 이 위에서 **리눅스 프로세스, 파이프, 시그널, 쉘 스크립트**를 이용하여 작은 ML 시스템을 구성하는, **시스템 소프트웨어 측면**을 구현한다.

---

## 2. 리포지터리 구조 및 UML 다이어그램

리포지터리의 기본 구조는 다음과 같다.

```text
.
├── include/
│   ├── backward_layer.hpp
│   ├── common.hpp
│   ├── forward_layer.hpp
│   ├── logger.hpp
│   ├── math_layer.hpp
│   ├── preprocess.hpp
│   └── trainer.hpp
├── src/
│   ├── backward_layer.cpp
│   ├── common.cpp
│   ├── forward_layer.cpp
│   ├── logger.cpp
│   ├── math_layer.cpp
│   ├── preprocess.cpp
│   └── trainer.cpp
├── data/
│   ├── train.csv
│   └── test.csv
├── uml/
│   ├── class.png
│   └── sequence.png
├── build.sh
└── run.sh
```

### 2.1. 아키텍처(클래스/컴포넌트) 다이어그램

전체 구성요소와 데이터 흐름을 요약한 UML 클래스/컴포넌트 다이어그램은 다음과 같다.

![Architecture overview](uml/class.png)

### 2.2. 실행 시퀀스(프로세스/데이터 흐름) 다이어그램

`run.sh`가 하나의 phase(`train` 혹은 `test`)를 수행할 때의 프로세스/파이프 흐름은 다음 UML 시퀀스 다이어그램과 같다.

![Execution sequence](uml/sequence.png)

---

## 3. 시스템 구조 요약

### 3.1. 실행 흐름 개요

1. 사용자는 최상위 디렉토리에서 `./run.sh`를 실행한다.
2. `run.sh`는 다음 세 단계에 대해 `trainer`를 순서대로 실행한다.

   * **pre-test**: `BACKWARD_MODE=test`로 `data/test.csv` 평가 (학습 전 성능 측정)
   * **train**: `BACKWARD_MODE=train`으로 `data/train.csv` 학습 및 파라미터 저장
   * **post-test**: `BACKWARD_MODE=test`로 `data/test.csv` 재평가 (학습 후 성능 측정)
3. 각 단계에서 `trainer`는 네 개의 자식 프로세스를 `fork` + `exec` 하여 파이프라인을 구성한다.

   * `preprocess` → `forward_layer` → `backward_layer` → `logger`
4. 각 단계의 실행 동안

   * `trainer`의 **stdout**은 `tee`를 통해 로그 파일(`logs/*.log`)과 터미널(progress bar)로 동시에 전달된다.
   * `trainer` 및 자식 프로세스의 **stderr**는 `logs/*.err`에 기록된다.
5. `logger`는 전체 입력 처리가 끝난 뒤,
   `SUMMARY <samples> <avg_loss> <avg_yhat>` 형식의 한 줄을 stdout에 출력하며,
   `run.sh`는 이를 파싱하여 각 phase에 대한 요약 메시지를 출력한다.

### 3.2. 파이프라인 단계별 역할

`trainer`는 각 phase에서 다음과 같은 프로세스 파이프라인을 구성한다.

```text
CSV (train.csv / test.csv)
        │
        ▼
preprocess     (bin/preprocess)
  - CSV 읽기
  - parse_csv_line()
  - normalize_sample()
        │
        │  (파이프: "id f0 f1 f2 f3 y")
        ▼
forward_layer  (bin/forward_layer)
  - parse_sample_line()
  - augment_features()
        │
        │  (파이프: "id f0' f1' f2' f3' y")
        ▼
backward_layer (bin/backward_layer)
  - parse_sample_line()
  - compute_forward()
  - (train 모드) compute_backward_and_update()
  - (test  모드) forward-only + loss 계산
        │
        │  (파이프: "id loss y_hat")
        ▼
logger         (bin/logger)
  - SAMPLE / SUMMARY 라인 출력
```

데이터 형식은 다음과 같이 정의된다.

* CSV 파일(`data/*.csv`)
  `id,f0,f1,f2,f3,y`
* 파이프라인 내부(프로세스 간 통신)
  `id f0 f1 f2 f3 y`
* `backward_layer` → `logger`
  `id loss y_hat`
* `logger` stdout

  * per-sample: `SAMPLE <id> LOSS <loss> YHAT <y_hat>`
  * summary: `SUMMARY <samples> <avg_loss> <avg_yhat>`

### 3.3. 모델 파라미터 저장 및 재사용

`math_layer` 내부에는 2-layer NN의 파라미터가 정적 전역 변수로 유지된다.
`backward_layer`는 환경 변수 `BACKWARD_MODE`에 따라 다음과 같이 동작한다.

* `BACKWARD_MODE=train` (학습 모드)

  * 입력 스트림 전체에 대해 forward + backward + SGD update를 수행한다.
  * 스트림이 종료되면 `math::save_parameters(MODEL_FILE)` 를 호출한다.
  * `MODEL_FILE`의 기본값은 `logs/model_params.txt`이다.

* `BACKWARD_MODE=test` (평가 모드)

  * 시작 시 `math::load_parameters(MODEL_FILE)` 를 시도한다.
  * 파일이 없으면 초기 파라미터를 그대로 사용한다.
  * forward-only 및 손실 (L = \frac{1}{2}(\hat{y} - y)^2) 계산을 수행하며, 파라미터 업데이트는 수행하지 않는다.

따라서 `./run.sh` 한 번의 실행은 다음 순서를 갖는다.

1. **pre-test**: 초기 파라미터 상태에서 `test.csv` 평가
2. **train**: `train.csv`로 학습 후 파라미터를 `logs/model_params.txt`에 저장
3. **post-test**: 저장된 파라미터를 로드하여 `test.csv` 재평가

---

## 4. 구현할 사항 개요

본 실습에서 수강생이 구현해야 할 사항은 크게 두 영역으로 구성된다.

1. **C++ 측 (프로세스/파이프/시그널)**

   * `src/trainer.cpp` 내부의 세 함수:

     * `sigchld_handler(int)`
     * `set_cloexec(int fd)`
     * `spawn_child(const char *prog, char *const argv[], int stdin_fd, int stdout_fd)`

2. **쉘 스크립트 측 (요약 출력 및 로그 파싱)**

   * `run.sh` 내부 `run_phase` 함수의 TODO 구간:

     * `SUMMARY` 라인 추출 및 파싱
     * 각 phase에 대한 요약 메시지 출력

그 외 파일(`math_layer`, `backward_layer`, `preprocess`, `forward_layer`, `logger`, `common` 등)은
**핵심 수학/모델 코드**로 간주하며, 인터페이스/기능의 변경 없이 사용하는 것을 원칙으로 한다.
디버깅을 위한 임시 출력 추가는 가능하지만, 최종 제출 시에는 정리하는 것이 바람직하다.

---

## 5. trainer.cpp에서의 구현 사항

아래는 `src/trainer.cpp` 중에서 수강생이 직접 구현해야 하는 세 함수의 스켈레톤이다.
각 함수는 프로세스 제어 및 파일 디스크립터 관리의 핵심 부분을 담당한다.

### 5.1. SIGCHLD 핸들러: `sigchld_handler(int)`

```cpp
volatile std::sig_atomic_t g_child_exited = 0;

void sigchld_handler(int)
{
    // TODO: mark that at least one child process has exited.
    // Hint: set the global flag g_child_exited to a non-zero value.
    //
    // This flag can be used in the main loop to notice that a SIGCHLD
    // was delivered and then call wait()/waitpid() to reap children.
}
```

* 목적: 자식 프로세스 종료 시 도착하는 `SIGCHLD` 시그널을 감지하고,
  전역 플래그 `g_child_exited`를 통해 이를 상위 로직에 알린다.
* 제약: 시그널 핸들러 내부에서는 async-signal-safe하지 않은 연산을 피해야 하므로,
  플래그 설정 외의 복잡한 작업은 수행하지 않는다.

### 5.2. FD_CLOEXEC 설정: `set_cloexec(int fd)`

```cpp
void set_cloexec(int fd)
{
    // TODO: mark this file descriptor as close-on-exec using fcntl().
    //
    // 1. Use fcntl(fd, F_GETFD) to get current flags.
    // 2. If the call succeeds, OR the result with FD_CLOEXEC.
    // 3. Use fcntl(fd, F_SETFD, new_flags) to update.
    //
    // This prevents child processes (after exec) from inheriting
    // these pipe descriptors unintentionally.
}
```

* 목적: `fd`에 `FD_CLOEXEC` 플래그를 설정하여,
  이후 `execvp()` 호출 시 해당 파일 디스크립터가 자동으로 닫히도록 한다.
* 구현 절차:

  1. `fcntl(fd, F_GETFD)`로 현재 플래그를 조회한다.
  2. 오류가 없으면 `flags | FD_CLOEXEC`로 새 플래그를 구성한다.
  3. `fcntl(fd, F_SETFD, new_flags)`로 갱신한다.

이를 통해, 필요하지 않은 파이프 디스크립터가 자식 프로세스들에 불필요하게 상속되는 것을 방지한다.

### 5.3. 자식 프로세스 생성: `spawn_child(...)`

```cpp
int spawn_child(const char *prog,
                char *const argv[],
                int stdin_fd,
                int stdout_fd)
{
    // TODO: fork a child process, hook up its stdin/stdout, and exec 'prog'.
    //
    // Required behavior:
    //   - Call fork().
    //   - On error (pid < 0), print an error with std::perror("fork")
    //     and return -1.
    //
    //   - In the child (pid == 0):
    //       * If stdin_fd >= 0 and stdin_fd != STDIN_FILENO,
    //         dup2(stdin_fd, STDIN_FILENO); on error, perror and _exit(1).
    //       * If stdout_fd >= 0 and stdout_fd != STDOUT_FILENO,
    //         dup2(stdout_fd, STDOUT_FILENO); on error, perror and _exit(1).
    //       * Call execvp(prog, argv).
    //         On error, print with std::perror("execvp") and _exit(1).
    //
    //   - In the parent:
    //       * Simply return the child's PID (the value from fork()).
    //
    // This function does NOT close any file descriptors; the caller
    // (trainer::run) remains responsible for closing unused pipe ends.
    return -1; // placeholder return to keep the skeleton compilable
}
```

* 목적: 프로그램 `prog`와 인자 `argv`를 실행하는 자식 프로세스를 생성하고,
  해당 자식의 표준입력/출력을 지정된 파이프 끝(`stdin_fd`, `stdout_fd`)에 연결한 뒤 `execvp()`를 호출한다.
* 자식 프로세스( `pid == 0` )에서는 다음을 수행해야 한다.

  * `stdin_fd`가 유효할 경우 `dup2(stdin_fd, STDIN_FILENO)` 호출.
  * `stdout_fd`가 유효할 경우 `dup2(stdout_fd, STDOUT_FILENO)` 호출.
  * 마지막으로 `execvp(prog, argv)`를 호출하여 실제 실행 파일로 교체.
  * 각 단계에서 오류 발생 시 `std::perror(...)` 후 `_exit(1)`로 종료.
* 부모 프로세스( `pid > 0` )에서는 자식 PID를 그대로 반환하며, FD 정리는 상위 로직(`trainer::run`)의 책임이다.

---

## 6. run.sh에서의 구현 사항

`run.sh`는 전체 실습을 수행하는 최상위 스크립트이다.
빌드/실행/진행률 출력 로직은 이미 포함되어 있으며, 수강생은 **요약 통계 출력 부분**을 완성해야 한다.

### 6.1. progress_bar (참고: 완성된 함수)

```bash
# --------------------------------------------------------------------
# progress_bar current total
#   - Render a simple textual progress bar on a single line.
#   - This function is fully implemented.
# --------------------------------------------------------------------
progress_bar() {
  local current=$1
  local total=$2
  local width=40

  if [[ "$total" -le 0 ]]; then
    printf "\r[progress] processing..."
    return
  fi

  local percent=$(( 100 * current / total ))
  local filled=$(( width * current / total ))
  local empty=$(( width - filled ))

  printf "\r["
  for ((i=0; i<filled; i++)); do printf "#"; done
  for ((i=0; i<empty; i++)); do printf "."; done
  printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}
```

해당 함수는 이미 구현되어 있으며, 수정 대상이 아니다.

### 6.2. run_phase의 TODO 구간

```bash
# --------------------------------------------------------------------
# run_phase phase csv log_file err_file
#
# Run one phase ("train" or "test"):
#   - Check that the CSV exists.
#   - Count non-empty lines to know how many samples to expect.
#   - Run trainer with BACKWARD_MODE set to the phase.
#   - Capture stdout into a log file and use SAMPLE lines for progress.
#   - After completion, extract a SUMMARY line and print a short summary.
# --------------------------------------------------------------------
run_phase() {
  local phase="$1"      # "train" or "test"
  local csv="$2"
  local log_file="$3"
  local err_file="$4"

  if [[ ! -f "$csv" ]]; then
    echo "[run] CSV file not found for phase '$phase': $csv" >&2
    return 1
  fi

  # Count total non-empty lines in the CSV.
  local total_lines
  total_lines=$(grep -cve '^\s*$' "$csv" || echo 0)

  echo "[run] Phase: $phase, file: $csv"
  echo "[run] Total lines: $total_lines"
  echo "[run] Logs: $log_file, errors: $err_file"

  local SAMPLES_PROCESSED=0

  # Pipeline:
  #   trainer stdout  -> tee (log_file + progress loop)
  #   trainer stderr  -> err_file
  #
  # The logger prints lines starting with "SAMPLE" for each sample.
  BACKWARD_MODE="$phase" "$TRAINER" "$csv" 2> "$err_file" | \
  tee "$log_file" | \
  while IFS= read -r line; do
    if [[ "$line" == SAMPLE* ]]; then
      SAMPLES_PROCESSED=$((SAMPLES_PROCESSED + 1))
      progress_bar "$SAMPLES_PROCESSED" "$total_lines"
    fi
  done

  echo

  # ------------------------------------------------------------------
  # TODO: Extract and print phase summary from "$log_file".
  #
  # The logger writes a final line of the form:
  #   SUMMARY <samples> <avg_loss> <avg_yhat>
  #
  # 1) Find the last such line in "$log_file" and store it in 'summary'.
  #    If no such line exists, 'summary' should be empty.
  #
  # 2) If 'summary' is non-empty:
  #      - Parse it into variables: samples, avg_loss, avg_yhat.
  #        (The first field is the literal word "SUMMARY".)
  #      - Print:
  #          [run] Phase '<phase>' summary: samples=<samples> avg_loss=<avg_loss> avg_yhat=<avg_yhat>
  #
  #    Otherwise:
  #      - Print:
  #          [run] Phase '<phase>' summary: (no SUMMARY line found)
  # ------------------------------------------------------------------
  local summary=""
  # TODO: summary=...

  if [[ -n "$summary" ]]; then
    local _tag samples avg_loss avg_yhat
    # TODO: read -r ...

    echo "[run] Phase '$phase' summary: samples=$samples avg_loss=$avg_loss avg_yhat=$avg_yhat"
  else
    echo "[run] Phase '$phase' summary: (no SUMMARY line found)"
  fi

  echo "[run] Phase '$phase' finished."
  echo "[run] Final logs: $log_file"
}
```

구현 내용은 다음과 같다.

1. `grep`과 `tail`을 이용하여 `SUMMARY`로 시작하는 마지막 줄을 `summary`에 저장한다.
2. `read -r`를 사용하여 `_tag samples avg_loss avg_yhat` 형식으로 분해한다.
3. 분해된 값을 이용해 지정된 형식의 요약 메시지를 출력한다.

예를 들어 올바르게 구현된 경우, phase 종료 시 다음과 같은 출력이 기대된다.

```text
[run] Phase 'test' summary: samples=1000 avg_loss=9.89868 avg_yhat=0.00509
```

---

## 7. 빌드 및 실행 방법

### 7.1. 빌드

```bash
./build.sh
```

* `bin/` 디렉토리에 다음 실행 파일이 생성된다.

  * `bin/preprocess`
  * `bin/forward_layer`
  * `bin/backward_layer`
  * `bin/logger`
  * `bin/trainer`
* 빌드 시 기존 `logs/` 디렉토리는 삭제 후 다시 생성된다.

### 7.2. 실행

```bash
./run.sh
```

* `data/train.csv`와 `data/test.csv`를 사용하여 다음 세 단계를 순차적으로 수행한다.

  1. pre-test (`BACKWARD_MODE=test`, 입력: `data/test.csv`)
  2. train (`BACKWARD_MODE=train`, 입력: `data/train.csv`)
  3. post-test (`BACKWARD_MODE=test`, 입력: `data/test.csv`)
* 각 단계에 대해:

  * 진행률(progress bar)이 터미널에 출력된다.
  * `logs/<phase>.log`에 stdout 로그가 저장된다.
  * `logs/<phase>.err`에 stderr 로그가 저장된다.
  * `SUMMARY` 정보를 기반으로 한 요약 메시지가 출력된다.

> 참고: 이전 실행에서 생성된 `logs/model_params.txt`가 남아 있는 경우,
> pre-test 단계가 “이미 학습된” 모델 파라미터로 수행될 수 있다.
> 완전히 초기 상태에서의 동작을 확인하고자 할 경우 `logs/` 디렉토리를 삭제한 뒤 다시 빌드/실행한다.

---

## 8. 제출 및 제한 사항

* **제출 형식**

  * GitHub에서 본 리포지터리를 fork한 뒤, 수정된 fork 리포지터리의 URL을 과제 제출란에 기입한다.
  * 또는 리포지터리를 내려받아 로컬에서 수정 후, 전체 디렉토리를 `zip`으로 압축하여 제출한다.

* **구현 범위**

  * `src/trainer.cpp`의 지정된 TODO 영역과 `run.sh`의 TODO 영역을 중심으로 구현한다.
  * `math_layer`, `backward_layer`, `preprocess`, `forward_layer`, `logger`, `common` 등의 코어 수치 연산 및 모델 관련 코드는 인터페이스/기능을 변경하지 않는다.

* **라이브러리 및 구조 변경 제한**

  * 표준 라이브러리 이외의 외부 라이브러리 추가는 허용하지 않는다.
  * 제공된 구조를 근본적으로 변경하는 것은 지양하며, TODO 영역 이외의 코드는 가능한 한 원형을 유지한다.

* **디버깅 출력**

  * 개발 과정에서 `std::cerr` 또는 `echo` 등을 이용한 디버깅 출력은 가능하나,
    최종 제출 시에는 필요하지 않은 디버깅 출력은 제거하는 것이 바람직하다.

---

## 9. 디버깅 및 확인을 위한 참고 사항

* `logs/*.err` 파일에는 다음과 같은 정보가 기록될 수 있다.

  * `backward_layer: loaded parameters from ...`
  * `backward_layer: saved parameters to ...`
  * `trainer: child <pid> exited with status ...`
  * `logger: failed to parse line: ...`
* 파이프라인의 각 단계를 부분적으로 확인하고자 할 경우, 개별 단계 실행이 가능하다.

  * 예:

    ```bash
    bin/preprocess data/train.csv | head
    ```
  * 또는:

    ```bash
    bin/preprocess data/train.csv | bin/forward_layer | head
    ```
* `logger` 프로세스에 `SIGUSR1` 시그널을 보내면, 현재까지의 통계 스냅샷을 stderr로 출력한다.

  * 예:

    ```bash
    pgrep logger   # logger PID 확인
    kill -USR1 <logger-pid>
    ```

본 실습의 목적은 **수학적 세부 구현**이 아니라,
**프로세스 생성, 파이프 연결, 시그널 처리, 쉘 스크립트에 의한 오케스트레이션**을 실제 코드로 구현하는 데 있습니다.
제공된 구조와 UML 다이어그램을 참조하여, 지정된 TODO 부분을 중심으로 구현을 완료하기 바랍니다!
