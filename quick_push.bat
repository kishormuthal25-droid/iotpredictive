@echo off
echo ========================================
echo  Quick Push to GitHub for Codespace
echo ========================================

cd /d "D:\IOT Predictive Maintenece System_copy\IOT Predictive Maintenece System"

echo Adding changes...
git add .

echo.
set /p commit_msg="Commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg="Update: Local changes for Codespace testing"

echo Committing...
git commit -m "%commit_msg%"

echo Pushing to GitHub...
git push origin master

echo.
echo ========================================
echo  SUCCESS! Code pushed to GitHub
echo  Now go to your Codespace and run:
echo  git pull origin master
echo ========================================
pause