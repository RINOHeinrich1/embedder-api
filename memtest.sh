#!/bin/bash

URL="http://localhost:8002/embed"
DATA='{"texts": ["Bonjour", "Test"]}'

N=10  # nombre total de requêtes

for i in $(seq 1 $N); do
  echo "[$(date '+%H:%M:%S')] Mémoire avant requête $i :"
  free -h | grep Mem

  # Lance la requête en arrière-plan
  curl -s -X POST $URL -H "Content-Type: application/json" -d "$DATA" &

  # Pour simuler un petit délai entre les requêtes (facultatif)
  sleep 0.5
done

wait
echo "Toutes les requêtes ont été envoyées."
