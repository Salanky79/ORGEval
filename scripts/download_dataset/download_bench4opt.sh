
#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO="${1:-ZiweiZhu0513/bench4opt}"
LOCAL_DIR="${2:-data/bench4opt}"

mkdir -p "${LOCAL_DIR}"

hf download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${LOCAL_DIR}"

echo "Bench4Opt downloaded to ${LOCAL_DIR}"
EOF

chmod +x scripts/download_bench4opt.sh