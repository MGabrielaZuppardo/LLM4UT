#!/bin/bash
# Checkout all buggy versions from d4j_fixed_project_list
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$PATH:/home/gabriela_zuppardo/defects4j/framework/bin

D4J_PROJ_BASE="/home/gabriela_zuppardo/defects4j/d4j_projects"
PROJECT_LIST="/mnt/d/LLM4UT/data/d4j_fixed_project_list"
LOG="/mnt/d/LLM4UT/tools/checkout_buggy.log"

total=$(wc -l < "$PROJECT_LIST")
count=0
skipped=0
failed=0

echo "Starting buggy checkout of $total projects at $(date)" | tee "$LOG"

while IFS= read -r line; do
    line="${line//$'\r'/}"       # strip CRLF
    line="${line%_fixed}"        # Chart_10
    project="${line%_*}"         # Chart
    num="${line##*_}"            # 10
    target_dir="$D4J_PROJ_BASE/${line}_buggy"

    count=$((count + 1))

    if [ -d "$target_dir" ]; then
        echo "[$count/$total] SKIP (exists): ${line}_buggy" | tee -a "$LOG"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[$count/$total] Checking out $project v${num}b ..." | tee -a "$LOG"
    defects4j checkout -p "$project" -v "${num}b" -w "$target_dir" >> "$LOG" 2>&1
    if [ $? -eq 0 ]; then
        # Create symlink: Chart_10/buggy -> Chart_10_buggy
        parent_dir="$D4J_PROJ_BASE/${line}"
        mkdir -p "$parent_dir"
        ln -sfn "$target_dir" "$parent_dir/buggy"
        echo "[$count/$total] OK: ${line}_buggy" | tee -a "$LOG"
    else
        echo "[$count/$total] FAILED: $project v${num}b" | tee -a "$LOG"
        failed=$((failed + 1))
    fi
done < "$PROJECT_LIST"

echo "" | tee -a "$LOG"
echo "Done at $(date). Total=$total, Skipped=$skipped, Failed=$failed" | tee -a "$LOG"
