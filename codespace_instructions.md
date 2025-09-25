# 🚀 Codespace Workflow Instructions

## Quick Development Cycle

### 1. Local Development (Claude Code)
- Make changes with Claude Code assistance
- Test basic functionality (if possible)

### 2. Push to GitHub
```bash
# Quick push script
quick_push.bat
```

### 3. In Codespace (4-8GB RAM)
```bash
# Pull latest changes
git pull origin master

# Install/update dependencies
pip install --upgrade -r requirements.txt

# Test system
python simple_test.py

# Launch IoT Dashboard
python launch_real_data_dashboard.py

# Run comprehensive tests
python tests/run_all_phase3_tests.py
```

### 4. Access Dashboard
- Codespace will auto-forward port 8060
- Click the popup to open dashboard
- Full NASA data with 97+ models available

## If Issues Found in Codespace

### Option A: Fix in Codespace
```bash
# Make changes directly
# Test changes
git add .
git commit -m "Fix: Issue description"
git push origin master
```

### Option B: Fix Locally (Recommended)
```bash
# In Codespace: Note the issue
# Switch to local Claude Code
# Fix the issue with AI assistance
# Push back using quick_push.bat
```

## Current System Status
- ✅ Repository: https://github.com/kishormuthal/iot-predictive-maintenance
- ✅ 360 files committed
- ✅ Codespace configuration ready
- ⚠️ Local dependency issues (will be fixed in Codespace)
- ✅ Memory-optimized configs available

## Codespace Advantages
- 4-8GB RAM vs 1.3GB local
- Clean Python environment
- All dependencies auto-install
- No local resource usage
- Persistent cloud storage