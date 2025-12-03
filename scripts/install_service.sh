sudo cp scripts/edge-attendance.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edge-attendance.service
sudo systemctl start edge-attendance.service