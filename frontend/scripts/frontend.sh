#!/bin/bash

# Frontend Process Manager
# Usage: ./frontend.sh {start|stop|restart|status}

PORT=3002
LOG_FILE="logs/frontend.log"
PID_FILE=".next/server.pid"

mkdir -p logs

get_pid() {
    # Try multiple ways to find the PID
    
    # 1. Check lsof (if available)
    if command -v lsof >/dev/null 2>&1; then
        PID=$(lsof -t -i:$PORT)
        if [ ! -z "$PID" ]; then
            echo $PID
            return
        fi
    fi
    
    # 2. Check netstat/ss (Linux)
    if command -v ss >/dev/null 2>&1; then
        PID=$(ss -lptn 'sport = :'$PORT | grep -o 'pid=[0-9]*' | cut -d= -f2)
        if [ ! -z "$PID" ]; then
            echo $PID
            return
        fi
    fi
    
    # 3. Check fuser
    if command -v fuser >/dev/null 2>&1; then
        PID=$(fuser $PORT/tcp 2>/dev/null)
        if [ ! -z "$PID" ]; then
            echo $PID
            return
        fi
    fi
}

start() {
    echo "Starting frontend on port $PORT..."
    PID=$(get_pid)
    if [ ! -z "$PID" ]; then
        echo "Frontend already running (PID: $PID). Stopping first..."
        stop
    fi
    
    nohup npm run start -- -p $PORT > $LOG_FILE 2>&1 &
    
    # Wait for startup
    echo "Waiting for startup..."
    for i in {1..30}; do
        if curl -s http://localhost:$PORT >/dev/null; then
            echo "Frontend is up!"
            return 0
        fi
        sleep 1
    done
    
    echo "Timeout waiting for frontend."
    return 1
}

stop() {
    PID=$(get_pid)
    if [ -z "$PID" ]; then
        echo "No running frontend found on port $PORT."
        return
    fi
    
    echo "Stopping frontend (PID: $PID)..."
    kill $PID 2>/dev/null
    
    # Wait for shutdown
    for i in {1..10}; do
        if [ -z "$(get_pid)" ]; then
            echo "Stopped."
            return
        fi
        sleep 1
    done
    
    echo "Force killing..."
    kill -9 $PID 2>/dev/null
}

status() {
    PID=$(get_pid)
    if [ ! -z "$PID" ]; then
        echo "Frontend running (PID: $PID)"
    else
        echo "Frontend stopped"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
