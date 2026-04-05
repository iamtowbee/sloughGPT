# Optional shortcuts — see README.md (Google Colab), scripts/run_colab_notebook_smoke.sh --help
.PHONY: help colab-smoke colab-test train-demo dev-stack

help:
	@echo "make dev-stack     API (:8000) + Next dev (:3000) — ./scripts/dev-stack.sh"
	@echo "make colab-smoke   ./scripts/run_colab_notebook_smoke.sh (needs pip install -e \".[notebook]\")"
	@echo "make colab-test    pytest tests/test_sloughgpt_colab_notebook.py -q"
	@echo "make train-demo    short local char-LM train (CPU; good first run after pip install -e .)"
	@echo "Run: ./scripts/run_colab_notebook_smoke.sh --help"

dev-stack:
	./scripts/dev-stack.sh

colab-smoke:
	./scripts/run_colab_notebook_smoke.sh

colab-test:
	python3 -m pytest tests/test_sloughgpt_colab_notebook.py -q

train-demo:
	python3 cli.py train --max-steps 5 --batch-size 4 --train-device cpu --no-mixed-precision --epochs 1 --log-interval 1 --eval-interval 99999
