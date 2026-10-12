#!/usr/bin/env bash
# S3: the inner task suite for one model — vLLM server (TP 4, bf16, thinking OFF) + lm-eval + EvalPlus.
#   bash tools/eval_suite.sh checkpoints/qwen3.8-27b base            # 📷 the summary table at the end
#   bash tools/eval_suite.sh checkpoints/bonsai2-27b-dense bonsai   # then: python tools/suite_summary.py runs/suite/bonsai --vs runs/suite/base
#   THINK=true GEN=16384 MAXLEN=32768 bash tools/eval_suite.sh …     # thinking mode (the final 27B run only)
#
# Tasks (all greedy, chat template, Table-8 group in brackets):
#   gsm8k_cot_llama 8-shot [Math] · math500 zero-shot boxed, math_verify grading [Math] · HumanEval+ via EvalPlus
#   [Coding] · ifeval [IF] · gpqa_diamond_cot_zeroshot [Knowledge] · mmlu_pro 5-shot CoT, 8 per category = 112 q [Knowledge]
# One-time: uv pip install -e ".[eval]"   (lm-eval[api,math,ifeval] + evalplus);  GPQA is gated → `huggingface-cli login`
# after accepting https://huggingface.co/datasets/Idavidrein/gpqa.  Each task runs in its own lm_eval call, so one
# failure (a gated dataset, a missing extra) costs one column, not the run.  ≈ 15 min per model on the box.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?model dir}"; TAG="${2:?tag, e.g. base | bonsai}"
TP="${TP:-4}"; PORT="${PORT:-8000}"; THINK="${THINK:-false}"; GEN="${GEN:-2048}"; MAXLEN="${MAXLEN:-8192}"
CONC="${CONC:-64}"; MMLU_PRO_PER_CAT="${MMLU_PRO_PER_CAT:-8}"
RUN="${RUN:-runs/suite/$TAG}"; mkdir -p "$RUN"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"      # lm-eval / evalplus clients want one; vLLM ignores it
API="http://127.0.0.1:$PORT/v1"
t0=$(date +%s)

if curl -s --max-time 2 "$API/models" >/dev/null 2>&1; then
  echo "something is already serving on :$PORT — stop it first (pkill -f 'vllm serve')"; exit 1
fi
echo "== serving $MODEL as '$TAG' (TP $TP, thinking $THINK, max_model_len $MAXLEN) → $RUN/vllm.log"
vllm serve "$MODEL" --served-model-name "$TAG" --tensor-parallel-size "$TP" --dtype bfloat16 \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 --seed 0 --port "$PORT" \
  --default-chat-template-kwargs "{\"enable_thinking\": $THINK}" > "$RUN/vllm.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  curl -s --max-time 2 "$API/models" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "vLLM died — see $RUN/vllm.log"; exit 1; }
  sleep 10
done
curl -s "$API/models" | python3 -c "import json,sys; print('   up:', [m['id'] for m in json.load(sys.stdin)['data']])"

python3 - <<'EOF' 2>/dev/null || true
import nltk; nltk.download("punkt_tab", quiet=True)      # ifeval's sentence counter
EOF

MARGS="model=$TAG,base_url=$API/chat/completions,num_concurrent=$CONC,max_retries=3,tokenized_requests=False"
run_task() {      # run_task <lm-eval task> [extra lm_eval flags…]
  local task="$1"; shift
  echo; echo "== $task  ($(( $(date +%s) - t0 )) s in)"
  lm_eval --model local-chat-completions --model_args "$MARGS" --tasks "$task" --apply_chat_template \
    --gen_kwargs "temperature=0,max_gen_toks=$GEN" --seed 0 --output_path "$RUN/lm_eval" --log_samples \
    --include_path tools/lm_tasks "$@" > "$RUN/$task.log" 2>&1 \
    && grep -E "^\|" "$RUN/$task.log" | tail -n +3 || { echo "   ✗ $task failed — tail of $RUN/$task.log:"; tail -5 "$RUN/$task.log"; }
}
run_task gsm8k_cot_llama --fewshot_as_multiturn
run_task math500
run_task ifeval
run_task gpqa_diamond_cot_zeroshot
run_task mmlu_pro --limit "$MMLU_PRO_PER_CAT"

echo; echo "== HumanEval+ (EvalPlus, greedy)  ($(( $(date +%s) - t0 )) s in)"
( cd "$RUN" && evalplus.evaluate --model "$TAG" --dataset humaneval --backend openai --base-url "$API" --greedy ) \
  > "$RUN/humaneval_plus.log" 2>&1 || echo "   ✗ evalplus failed — tail of $RUN/humaneval_plus.log:"
grep -E "humaneval|pass@1" "$RUN/humaneval_plus.log" | tail -4

echo; echo "== summary  📷   ($(( $(date +%s) - t0 )) s total)"
python3 tools/suite_summary.py "$RUN" ${VS:+--vs "$VS"}
