#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run the whole experiment from Thor with the gpu-rent `gpu` CLI (profile trtexact).
# 1. Rent an A100-SXM4-40GB (retry until RunPod has one), build the engines, run them, pull, delete.
# 2. Rent an A100-SXM4-80GB, push the 40GB engines, run them, pull, delete.
# 3. Compare. Start detached so it survives the shell:
#    setsid nohup experiments/trt-bit-exact-a100/pipeline.sh > ~/trt-exact-work/pipeline.log 2>&1 &
set -uo pipefail

WORK=~/trt-exact-work
RESULTS=$WORK/results
HERE=$(cd "$(dirname "$0")" && pwd)
RETRY_S=90
POD=""
LAST=""

log() { echo "$(date '+%F %T') $*"; }
cleanup() { [ -n "$POD" ] && { log "deleting pod $POD"; gpu down "$POD" --no-pull; }; }
trap cleanup EXIT

# Rent a pod; retry while RunPod has no instance. Sets POD.
rent() {
    local gpu=$1 try=0 out
    while true; do
        try=$((try + 1))
        out=$(gpu up trtexact --gpu "$gpu" --hours 1 --yes 2>&1)
        POD=$(echo "$out" | sed -n 's/^created pod \([a-z0-9]*\).*/\1/p')
        if echo "$out" | grep -q '^ready:'; then
            log "RENTED $gpu pod $POD after $try tries"
            return 0
        fi
        POD=""  # `gpu up` deletes a pod that failed to start
        echo "$out" | grep -q 'no instances currently available' \
            || log "try $try: $(echo "$out" | tail -2 | tr '\n' ' ')"
        [ $((try % 20)) -eq 0 ] && log "still no $gpu after $try tries"
        sleep $RETRY_S
    done
}

# Run on the pod, pull, delete, copy outputs for <tag> to $RESULTS.
run_and_pull() {
    local tag=$1 cmd=$2
    local a
    for a in 1 2 3; do  # pods sometimes drop SSH for a while right after setup
        gpu run --pod "$POD" -- "$cmd" && break
        [ $a -eq 3 ] && { log "FAILED on $tag"; exit 1; }
        log "run on $tag failed (try $a); retrying in 60 s"
        sleep 60
    done
    gpu down "$POD" || exit 1
    local got=~/gpu-results/$POD/root/work
    LAST=$POD POD=""
    mkdir -p "$RESULTS"
    cp -r "$got/out/$tag" "$RESULTS/" || exit 1
    log "DONE $tag -> $RESULTS/$tag"
}

log "START"
if [ -f "$RESULTS/a100_40gb/meta.json" ]; then
    log "40GB results exist; skipping the build"
else
    rent A100_40
    run_and_pull a100_40gb "python run_trt.py build /root/work && python run_trt.py run /root/work a100_40gb"
    mkdir -p "$WORK/engines"
    cp ~/gpu-results/$LAST/root/work/engines/*.engine "$WORK/engines/"  # pushed to the 80GB pod
fi

rent A100
run_and_pull a100_80gb "python run_trt.py run /root/work a100_80gb"

~/.venvs/mo/bin/python "$HERE/compare.py" "$RESULTS" | tee "$RESULTS/compare.md"
log "FINISHED"
