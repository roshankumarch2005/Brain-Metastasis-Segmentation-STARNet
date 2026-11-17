#!/bin/bash
# start_web_server.sh
# --------------------
# Starts or restarts the 3D visualization web server on port 8080 in a background tmux session.

DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
PORT=8080

if tmux has-session -t web_server 2>/dev/null; then
    echo "[✓] Web server is already active in background tmux session 'web_server' on port $PORT!"
else
    tmux new-session -d -s web_server "python3 -m http.server $PORT --directory $DIR"
    echo "[🚀] Web server started successfully in background tmux session 'web_server' on port $PORT!"
fi

echo ""
echo "Available 3D Portals:"
echo " • Model B0 (Vanilla):    http://localhost:8080/results/figures_saluja_b0_html/index.html"
echo " • Model B1 (SIAM+BAA):   http://localhost:8080/results/figures_saluja_b1_html/index.html"
echo " • Model B3 (Boundary):   http://localhost:8080/results/figures_saluja_b3_html/index.html"
echo ""
echo "To stop the server at any time: tmux kill-session -t web_server"
