@echo off
:: Arrêter le serveur Flask
taskkill /F /IM python.exe /T

:: Reconstruire et redémarrer l'application Flask
cd .. && cargo build --release && cd src && python app.py
