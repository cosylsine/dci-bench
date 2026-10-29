#!/usr/bin/env bash
# Pinned raw sources for the 2026-10-14 DCI expansion. Safe to resume.
set -euo pipefail
DCI_EXPANSION_ROOT=${DCI_EXPANSION_ROOT:-/mnt/afs2/202609/embedding_models/dci-bench}
HFD=${HFD:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/hfd.sh}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# Keep the active proxy by default. Set DCI_KEEP_PROXY=0 only after verifying
# a current proxy failure; an old connection failure does not justify bypassing it.
if [[ ${DCI_KEEP_PROXY:-1} != 1 ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
fi
for dependency in curl jq aria2c; do
  command -v "$dependency" >/dev/null || { echo "Missing dependency: $dependency" >&2; exit 2; }
done
# hfd.sh is used unchanged. Bound metadata retries and avoid IPv6 connection stalls.
curl() { command curl -4 --connect-timeout 15 --max-time 90 --retry 3 --retry-delay 2 "$@"; }
export DCI_HFD_HTTP2_HELPER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/hfd_http2_transport.py"
aria2c() {
  if [[ ${DCI_HFD_HTTP2:-0} == 1 ]]; then
    python "$DCI_HFD_HTTP2_HELPER" "$@"
  else
    command aria2c --disable-ipv6=true --connect-timeout=15 --timeout=60 --max-tries=6 --retry-wait=3 "$@"
  fi
}
# POSIX awk's default numeric formatting breaks hfd's progress totals above 2GB.
awk() { command awk -v OFMT='%.0f' -v CONVFMT='%.17g' "$@"; }
export -f curl aria2c awk
download() {
  local track=$1 name=$2 repo=$3 revision=$4
  shift 4
  bash "$HFD" "$repo" --dataset --revision "$revision" \
    --local-dir "$DCI_EXPANSION_ROOT/$track/$name/raw" --tool aria2c -x 4 -j 8 --include "$@"
}
case ${1:-all} in
  wixqa)
    download eval_data wixqa Wix/WixQA d662dc42479c14e202eccd832f8c4b66a035c4cc \
      README.md 'wix_kb_corpus/*' 'wixqa_expertwritten/*' 'wixqa_simulated/*' ;;
  browsecomp)
    download eval_data browsecomp_plus Tevatron/browsecomp-plus-corpus b27b02bc3e45511b8b82a13e6f90ce761df726f6 README.md 'data/*' ;;
  browsecomp_queries)
    download eval_data browsecomp_plus_queries Tevatron/browsecomp-plus 144cff8e35b5eaef7e526346aa60774a9deb941f README.md 'data/*' ;;
  enterprise)
    download eval_data enterprise_rag onyx-dot-app/EnterpriseRAG-Bench 69916e31c68aa5963c00248fd7f0bc12d04fd235 README.md 'data/*' ;;
  worksurface)
    download eval_data worksurface lhpku20010120/WorkSurface-Bench 3b3aef45f1e8f69416d6112c46e6e873cf522315 \
      README.md data/tasks.jsonl 'resources/profiles/*/kb_docs/*' ;;
  arb)
    download eval_data_code agent_retrieval_bench eyuansu71/agent_retrieval_bench 5901e1ee3aff048290db72edf9c63bc498b79ea3 \
      README.md 'releases/v2_code2test/*' 'releases/v2_comment2context/*' 'releases/v2_trace2code/*' ;;
  all)
    for source in wixqa browsecomp browsecomp_queries enterprise arb; do bash "$0" "$source"; done ;;
  *) echo "Usage: $0 [all|wixqa|browsecomp|browsecomp_queries|enterprise|worksurface|arb]" >&2; exit 2 ;;
esac
