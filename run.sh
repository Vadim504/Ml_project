#!/bin/bash

PORT=8000

# 1. Ищем PID процесса, который занимает порт, и убиваем его (если он есть)
echo "Очистка порта $PORT..."
PID=$(lsof -t -i:$PORT)

if [ -z "$PID" ]; then
    echo "Порт свободен."
else
    echo "Убиваем процесс $PID на порту $PORT"
    kill -9 $PID
fi

# 2. Запускаем uvicorn
echo "Запуск сервера..."
uvicorn backend.main:app --reload --port $PORT