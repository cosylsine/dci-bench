.PHONY: test prepare phase1-smoke phase2-toy phase3-one phase4-task phase5-all phase5-sglang

TASK ?= LLMPublicHealthQA
QUERY_ID ?= Q25
QUERY_IDS ?= Q71 Q72
MODEL_KEY ?= MiniCPM5-2B
BACKEND_MANIFEST ?=

test:
	python -m unittest discover -s tests -v

prepare:
	python -m dci_bench.data.workspace_builder --task $(TASK) --skip-existing --sample-queries 5

phase1-smoke: prepare test

phase2-toy:
	python scripts/phase2_toy_smoke.py

phase3-one:
	python scripts/phase3_single_sample.py --task $(TASK) --query-id $(QUERY_ID) --model-key $(MODEL_KEY) --backend-manifest $(BACKEND_MANIFEST)

phase4-task:
	python scripts/run_task.py --model-key $(MODEL_KEY) --task $(TASK) --query-ids $(QUERY_IDS) --backend-manifest $(BACKEND_MANIFEST)

phase5-all:
	python scripts/run_all.py --model-key $(MODEL_KEY) --backend-manifest $(BACKEND_MANIFEST)

phase5-sglang:
	scripts/run_phase5_with_sglang.sh
