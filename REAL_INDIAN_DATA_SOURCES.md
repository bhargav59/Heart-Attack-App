# Real Indian Medical Data Sources

## 🎯 Summary

This document lists **authentic Indian origin medical datasets** for heart disease research, as opposed to synthetic/simulated data.

---

## ⚡ Quick Access - Best Options

### 1. **Z-Alizadeh Sani Dataset** (RECOMMENDED - Real Asian Data)
- **URL**: https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani
- **Records**: 303 patients
- **Features**: 54 (demographics, symptoms, ECG, echo, lab results)
- **Source**: Real hospital data from coronary angiography
- **Authentic**: ✅ Yes - Published medical study
- **Access**: Free download

### 2. **Mendeley Data - Indian CVD Datasets**
- **URL**: https://data.mendeley.com/
- **Search**: "India cardiovascular disease" OR "Indian heart disease"
- **Description**: Academic researchers publish Indian hospital datasets
- **Authentic**: ✅ Yes - Peer-reviewed research
- **Access**: Free with registration

### 3. **ICMR-INDIAB Study**
- **URL**: https://www.icmr.gov.in/
- **Description**: Largest Indian diabetes & CVD population study
- **Records**: 100,000+ participants across India
- **Authentic**: ✅ Yes - Government research
- **Access**: Request from ICMR

---

## 🏥 Indian Medical Institutions (Request Access)

### Premier Medical Institutes

| Institution | Department | Contact/URL | Data Type |
|------------|------------|-------------|-----------|
| **AIIMS Delhi** | Cardiology | https://www.aiims.edu/en/departments/cardiology.html | Real patient records |
| **PGI Chandigarh** | Cardiology | https://pgimer.edu.in/ | Hospital database |
| **CMC Vellore** | Cardiology | https://www.cmch-vellore.edu/ | Clinical trials data |
| **Sree Chitra Tirunal Institute** | CVD Research | https://sctimst.ac.in/ | Specialized CVD data |
| **Narayana Health** | Cardiac Care | Corporate partnership | Large patient database |
| **Apollo Hospitals** | Cardiology | Corporate partnership | Multi-center data |

### How to Request Access:
1. Draft research proposal
2. Email cardiology/research departments
3. Explain research purpose and ethical safeguards
4. Sign data sharing agreements
5. Obtain IRB/Ethics Committee approval

**Email Template**: See `data/search_authentic_indian_data.py`

---

## 📊 Indian Government Health Data

### National Health Databases

| Source | URL | Description | Access |
|--------|-----|-------------|--------|
| **ICMR** | https://www.icmr.gov.in/ | National health research database | Request access |
| **NFHS-5** | http://rchiips.org/nfhs/ | National Family Health Survey | Free download |
| **NHM** | https://nhm.gov.in/ | National Health Mission data | Public portal |
| **India Data Portal** | https://data.gov.in/ | Government open data | Free download |

### Key Datasets Available:

1. **ICMR-INDIAB**
   - Indian Diabetes study with CVD outcomes
   - 100,000+ participants
   - Urban and rural populations

2. **National NCD Monitoring Survey**
   - Non-communicable disease surveillance
   - Includes cardiovascular risk factors
   - State-wise data

3. **India Hypertension Control Initiative**
   - Blood pressure and heart disease data
   - Primary health center records

---

## 🔬 Published Research Studies with Indian Data

### Major Studies

| Study Name | Description | Records | Access |
|------------|-------------|---------|--------|
| **PURE Study (India)** | Prospective Urban Rural Epidemiology | ~20,000 | Publications |
| **CURES Study** | Chennai Urban Rural Epidemiology Study | Large cohort | Dr. Mohan's Centre |
| **Jaipur Heart Watch** | Population-based CVD study | Regional | PubMed papers |
| **India Heart Study** | Multi-center CVD registry | Variable | Cardiological Society |

### How to Access:
1. Search PubMed for study name + "India"
2. Check "Data Availability" section in papers
3. Download supplementary materials
4. Contact corresponding authors
5. Request dataset access for research

---

## 💾 Verified Kaggle Datasets

### Search Strategy:
- **URL**: https://www.kaggle.com/search?q=heart+disease+india
- **Filters**: Sort by "Most Votes" and check metadata

### Verification Checklist:
✅ Look for hospital/institution name  
✅ Check publication references  
✅ Verify author credentials (medical professional?)  
✅ Read comments for authenticity discussions  
✅ Check dataset description for source citation  
✅ Look for ethics committee approval  

### Warning Signs (Synthetic Data):
❌ No source institution listed  
❌ Created by "data scientist" without medical affiliation  
❌ Too many records (>10,000) without hospital source  
❌ Perfect distributions  
❌ All correlations < 0.05  

---

## 🔍 Data Quality Indicators

### ✅ REAL Medical Data Shows:

| Indicator | Expected Value | Why It Matters |
|-----------|---------------|----------------|
| Age-Risk Correlation | > 0.3 | Age is strongest CVD predictor |
| BP-Risk Correlation | > 0.2 | Hypertension is major risk factor |
| Cholesterol-Risk | > 0.15 | Established medical relationship |
| ROC AUC | > 0.70 | Good predictive performance |
| Missing Values | Some present | Real data has gaps |
| Value Distributions | Skewed/realistic | Medical measurements aren't uniform |

### ❌ SYNTHETIC Data Shows:

| Indicator | Observed Value | Red Flag |
|-----------|---------------|----------|
| All Correlations | < 0.05 | No real patterns |
| ROC AUC | ≈ 0.50 | Random guessing |
| Missing Values | None | Too perfect |
| Distributions | Uniform/normal | Unrealistic |
| Source | "Kaggle user" | Not medical institution |
| Validation | None | No clinical studies |

**Your Current Dataset**: Max correlation 0.021 → **SYNTHETIC** ❌

---

## 📥 Download Instructions

### Z-Alizadeh Sani Dataset (Recommended)

```bash
# 1. Visit UCI Repository
open https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani

# 2. Download ZIP file
# (Click "Download" button on page)

# 3. Extract to project
unzip z_alizadeh_sani.zip -d data/real_datasets/z_alizadeh_sani/

# 4. Process dataset
python data/process_z_alizadeh_sani.py
```

### ICMR-INDIAB Dataset

```bash
# 1. Email request to ICMR
# Subject: Request for ICMR-INDIAB Dataset Access

# 2. Include:
# - Research proposal
# - Institutional affiliation
# - Data usage agreement
# - Ethics approval (if required)

# 3. Wait for approval (2-4 weeks)

# 4. Download via provided link
```

---

## 📧 Email Template for Data Requests

```
Subject: Request for Access to Cardiovascular Disease Dataset for Research

Dear [Department Head / Research Coordinator],

I am a researcher working on developing machine learning models for heart 
attack risk prediction. I am currently seeking authentic Indian medical data 
to improve model accuracy and ensure clinical relevance for Indian populations.

I would greatly appreciate access to anonymized cardiovascular disease patient 
data from your institution, including:
- Demographic information (age, gender)
- Clinical parameters (BP, cholesterol, diabetes history)
- Cardiovascular outcomes

The data will be used solely for academic research purposes and all patient 
information will remain confidential. I am willing to:
- Sign data sharing agreements
- Obtain IRB/Ethics Committee approval
- Acknowledge your institution in publications
- Share research findings with your team

Please let me know the process for requesting access to such datasets.

Thank you for your consideration.

Best regards,
[Your Name]
[Your Institution]
[Your Contact Information]
```

---

## 🚀 Immediate Action Plan

### Week 1: Free Downloads
- [ ] Download Z-Alizadeh Sani dataset from UCI
- [ ] Search Mendeley Data for Indian CVD datasets
- [ ] Download NFHS-5 survey data
- [ ] Search data.gov.in for cardiovascular data

### Week 2: Requests
- [ ] Email ICMR for INDIAB dataset access
- [ ] Contact AIIMS Delhi cardiology research department
- [ ] Request data from PGI Chandigarh
- [ ] Search PubMed for Indian CVD studies with supplementary data

### Week 3: Processing
- [ ] Process downloaded datasets
- [ ] Map features to your schema
- [ ] Analyze data quality (correlations, ROC AUC)
- [ ] Compare with synthetic dataset

### Week 4: Retraining
- [ ] Retrain model on real Indian data
- [ ] Evaluate improvement in accuracy
- [ ] Update API with new model
- [ ] Document results

---

## 📚 Additional Resources

### Research Databases
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/ (Search: "heart disease India dataset")
- **IEEE Xplore**: https://ieeexplore.ieee.org/ (Medical datasets)
- **Google Scholar**: https://scholar.google.com/ (Academic papers with data)

### Medical Societies
- **Cardiological Society of India**: https://www.csi.org.in/
- **Indian Medical Association**: https://www.ima-india.org/
- **Indian Heart Association**: Contact for research collaborations

### Data Repositories
- **Mendeley Data**: https://data.mendeley.com/
- **Figshare**: https://figshare.com/
- **Zenodo**: https://zenodo.org/
- **Dryad**: https://datadryad.org/

---

## ⚠️ Important Notes

1. **Ethics & Privacy**
   - Always use anonymized data
   - Respect data usage agreements
   - Obtain necessary approvals
   - Cite data sources properly

2. **Data Quality**
   - Verify authenticity before training
   - Check correlations (should be > 0.2 for key features)
   - Validate with medical literature
   - Test on holdout set

3. **Expected Improvements**
   - Real data: 75-85% accuracy (vs current 69%)
   - Strong correlations: Age > 0.3, BP > 0.2
   - ROC AUC > 0.75 (vs current 0.48)
   - Clinical validation possible

---

## 📞 Contact Information

### For Questions:
- ICMR: icmrhqds@gmail.com
- AIIMS: aiimsnewdelhi@gmail.com
- Data.gov.in: data-india@nic.in

### For Collaborations:
Contact cardiology departments at major Indian hospitals for research partnerships.

---

**Last Updated**: November 11, 2025  
**Status**: Ready for data acquisition  
**Next Step**: Download Z-Alizadeh Sani dataset or request ICMR-INDIAB access
