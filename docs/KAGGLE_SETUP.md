# 🔑 Kaggle Setup Guide

Complete guide for setting up Kaggle API access to download the Walmart dataset for retail sales forecasting.

## 🎯 Quick Setup

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Download API key from Kaggle
# 3. Place kaggle.json in ~/.kaggle/
# 4. Set permissions
chmod 600 ~/.kaggle/kaggle.json

# 5. Test connection
kaggle competitions list
```

## 📋 Prerequisites

- **Kaggle Account**: Free registration at [kaggle.com](https://kaggle.com)
- **Python Environment**: Python 3.7+ with pip
- **Competition Access**: Join Walmart competition (explained below)

## 🔧 Step-by-Step Setup

### **Step 1: Create Kaggle Account**

1. Visit [kaggle.com](https://kaggle.com)
2. Click **"Register"** and create free account
3. Verify email address
4. Complete profile setup (recommended)

### **Step 2: Generate API Key**

1. **Login to Kaggle**
2. **Navigate to Account Settings**:
   - Click your profile picture (top-right)
   - Select **"Account"**
   
3. **Generate API Token**:
   - Scroll to **"API"** section
   - Click **"Create New API Token"**
   - Download `kaggle.json` file

### **Step 3: Install Kaggle CLI**

```bash
# Option 1: Using pip
pip install kaggle

# Option 2: Using Poetry (recommended for this project)
poetry add kaggle

# Option 3: Using conda
conda install -c conda-forge kaggle
```

### **Step 4: Configure API Key**

#### **macOS/Linux Setup**
```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Move downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set secure permissions (IMPORTANT!)
chmod 600 ~/.kaggle/kaggle.json
```

#### **Windows Setup**
```powershell
# Create Kaggle directory
mkdir $env:USERPROFILE\.kaggle

# Move downloaded file
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\

# Windows automatically sets appropriate permissions
```

### **Step 5: Verify Setup**

```bash
# Test API connection
kaggle competitions list

# Expected output: List of active competitions
# If you see this, setup is successful! ✅
```

## 🏪 Walmart Competition Access

### **Join the Competition**

1. **Navigate to Competition**:
   - Visit: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
   
2. **Accept Rules**:
   - Click **"Join Competition"**
   - Read and accept competition rules
   - Click **"I Understand and Accept"**

3. **Verify Access**:
   ```bash
   kaggle competitions download -c walmart-recruiting-store-sales-forecasting
   ```

### **Important Notes**
- ⚠️ **Must accept rules** before downloading data
- ⚠️ **Competition is closed** but data remains accessible
- ✅ **No submission required** for data access
- ✅ **Free access** to historical data

## 📁 Data Download

### **Automatic Download (Recommended)**
```bash
# Navigate to project directory
cd projects/retail_sales_walmart

# Run download script
python scripts/download_data.py

# Files will be saved to data/raw/
```

### **Manual Download**
```bash
# Download competition files
kaggle competitions download -c walmart-recruiting-store-sales-forecasting

# Extract to project directory
unzip walmart-recruiting-store-sales-forecasting.zip -d projects/retail_sales_walmart/data/raw/

# Clean up zip file
rm walmart-recruiting-store-sales-forecasting.zip
```

### **Verify Downloaded Files**
```bash
ls -la projects/retail_sales_walmart/data/raw/

# Expected files:
# train.csv          (421,570 rows, ~28MB)
# test.csv           (115,064 rows, ~8MB)  
# stores.csv         (45 rows, ~1KB)
# features.csv       (8,190 rows, ~500KB)
# sampleSubmission.csv
```

## 🔍 Troubleshooting

### **Common API Issues**

#### **"Kaggle API key not found"**
```bash
# Check file location
ls -la ~/.kaggle/kaggle.json

# If missing, re-download from Kaggle account settings
# Ensure file is in correct location with proper permissions
```

#### **"401 Unauthorized"**
```bash
# Regenerate API key
# 1. Go to Kaggle account settings
# 2. Click "Create New API Token"  
# 3. Replace old kaggle.json file
# 4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
```

#### **"403 Forbidden - Competition data not accessible"**
```bash
# Solution: Accept competition rules
# 1. Visit competition page
# 2. Click "Join Competition"
# 3. Accept rules and terms
# 4. Retry download
```

### **Network and Connection Issues**

#### **"Connection timeout"**
```bash
# Increase timeout
kaggle competitions download -c walmart-recruiting-store-sales-forecasting --timeout 300

# Use proxy if behind corporate firewall
kaggle config set proxy_address "http://proxy.company.com:8080"
```

#### **"SSL certificate verify failed"**
```bash
# Update certificates
pip install --upgrade certifi

# Or disable SSL verification (not recommended)
kaggle config set ssl_ca_cert ""
```

### **Permission Issues**

#### **macOS/Linux: "Permission denied"**
```bash
# Fix kaggle.json permissions
chmod 600 ~/.kaggle/kaggle.json

# Ensure Kaggle directory is accessible
chmod 755 ~/.kaggle
```

#### **Windows: "Access denied"**
```powershell
# Run PowerShell as Administrator
# Check file location
Get-ChildItem $env:USERPROFILE\.kaggle\

# Reset permissions if needed
icacls $env:USERPROFILE\.kaggle\kaggle.json /reset
```

## ⚙️ Advanced Configuration

### **Environment Variables**
```bash
# Alternative to kaggle.json file
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Verify
kaggle competitions list
```

### **Proxy Configuration**
```bash
# Configure proxy settings
kaggle config set proxy_address "http://proxy.example.com:8080"
kaggle config set proxy_username "proxy_user"
kaggle config set proxy_password "proxy_pass"
```

### **Custom Download Location**
```bash
# Download to specific directory
kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p /custom/path/

# Set default download path
kaggle config set path "/custom/default/path"
```

## 🔐 Security Best Practices

### **API Key Security**
- ✅ **Keep API key private** - never commit to version control
- ✅ **Use secure permissions** (600) on kaggle.json
- ✅ **Regenerate regularly** - especially after sharing systems
- ❌ **Don't share** API keys in code or documentation
- ❌ **Don't email** or message API keys

### **Team Collaboration**
```bash
# Add to .gitignore (already included in this project)
echo "kaggle.json" >> .gitignore

# For teams: Each member needs own API key
# Don't share a single key across team members
```

## 📊 Data Usage Guidelines

### **Competition Data License**
- ✅ **Educational use** permitted
- ✅ **Research and analysis** allowed  
- ✅ **Portfolio projects** acceptable
- ❌ **Commercial redistribution** prohibited
- ❌ **Claiming ownership** of data not allowed

### **Citation Requirements**
```
Walmart Store Sales Forecasting Dataset
Source: Kaggle Competition
URL: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
Accessed: [Date]
```

## 🔄 Maintenance

### **Regular Tasks**
- **Monthly**: Verify API key still active
- **Quarterly**: Check for Kaggle CLI updates
- **As needed**: Regenerate API key if compromised

### **Updates**
```bash
# Update Kaggle CLI
pip install --upgrade kaggle

# Check version
kaggle --version
```

## 🔗 Related Documentation

- **[Data Download Guide](DATA_DOWNLOAD_README.md)**: Complete data setup
- **[Dataset Status Report](DATASET_STATUS.md)**: Current data availability
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Common fixes
- **[Project Structure Guide](PROJECT_STRUCTURE.md)**: Folder organization

## 📞 Support

### **Kaggle Support**
- **Documentation**: https://kaggle.com/docs/api
- **Forums**: https://kaggle.com/discussions
- **Contact**: Use Kaggle support form for account issues

### **Project Support**
- **Data Issues**: Check [Dataset Status](DATASET_STATUS.md)
- **API Problems**: Review this guide's troubleshooting section
- **Technical Issues**: See [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*Last updated: October 2025 | Next review: January 2026*