#!/bin/bash
# Ollama Watchdog — 크래시 시 자동 재시작
# Usage: ./ollama_watchdog.sh &

export OLLAMA_MODELS=/var/ollama/models

LOGFILE="/tmp/ollama_watchdog.log"
CHECK_INTERVAL=10
MAX_RESTART_ATTEMPTS=5
RESTART_COUNT=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

check_ollama() {
    curl -sf http://localhost:11434/ > /dev/null 2>&1
}

start_ollama() {
    log "Ollama 시작 중..."
    OLLAMA_MODELS=/var/ollama/models nohup ollama serve >> /tmp/ollama_serve.log 2>&1 &
    sleep 5
    if check_ollama; then
        log "Ollama 시작 완료 (PID: $(pgrep -f 'ollama serve'))"
        RESTART_COUNT=0
        return 0
    else
        log "Ollama 시작 실패"
        return 1
    fi
}

log "=== Ollama Watchdog 시작 ==="

while true; do
    if ! check_ollama; then
        log "⚠ Ollama 무응답 감지"
        RESTART_COUNT=$((RESTART_COUNT + 1))

        if [ $RESTART_COUNT -gt $MAX_RESTART_ATTEMPTS ]; then
            log "❌ 최대 재시작 횟수 ($MAX_RESTART_ATTEMPTS) 초과 — watchdog 종료"
            exit 1
        fi

        # 기존 프로세스 정리
        pkill -f "ollama serve" 2>/dev/null
        sleep 3

        start_ollama
    fi

    sleep $CHECK_INTERVAL
done
