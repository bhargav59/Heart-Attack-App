"""
Download Z-Alizadeh Sani Dataset from UCI ML Repository
Real medical data: 303 Asian patients with coronary artery disease
"""
import requests
from pathlib import Path
import zipfile
import io

def download_z_alizadeh_sani():
    """Download the Z-Alizadeh Sani dataset from UCI"""
    print("=" * 80)
    print("DOWNLOADING Z-ALIZADEH SANI DATASET")
    print("=" * 80)
    print("\n📋 Dataset Information:")
    print("   Source: UCI Machine Learning Repository")
    print("   Patients: 303 (Asian population)")
    print("   Features: 54 clinical features")
    print("   Type: REAL medical data from coronary angiography")
    print("   Published: 2013")
    print()
    
    # Create output directory
    output_dir = Path("data/real_datasets/z_alizadeh_sani")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UCI dataset URL
    base_url = "https://archive.ics.uci.edu/static/public/412"
    dataset_url = f"{base_url}/z+alizadeh+sani.zip"
    
    print(f"🔗 Downloading from: {dataset_url}")
    print("⏳ Please wait...\n")
    
    try:
        # Download the ZIP file
        response = requests.get(dataset_url, timeout=60, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"📦 File size: {total_size / 1024:.2f} KB")
        
        # Read the ZIP content
        zip_content = io.BytesIO(response.content)
        
        # Extract the ZIP file
        print("📂 Extracting files...")
        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(output_dir)
        
        print(f"✅ Downloaded and extracted to: {output_dir}")
        
        # List extracted files
        print("\n📁 Extracted files:")
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                size = file.stat().st_size
                print(f"   - {file.name} ({size:,} bytes)")
        
        print("\n" + "=" * 80)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 80)
        print("""
Next Steps:
1. Examine the dataset files
2. Read the documentation/README
3. Process and map features to your schema
4. Train model on real medical data
5. Compare results with synthetic dataset

Expected improvements with real data:
- Stronger correlations (Age-Risk > 0.3)
- Better ROC AUC (> 0.75)
- Higher accuracy (75-85%)
- Clinical validation possible
        """)
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        print("\nAlternative download methods:")
        print("1. Manual download:")
        print(f"   Visit: https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani")
        print("   Click 'Download' button")
        print(f"   Extract to: {output_dir}")
        print("\n2. Try different mirror:")
        print("   The UCI repository may have moved or be temporarily down")
        return False
    
    except zipfile.BadZipFile:
        print("\n❌ Downloaded file is not a valid ZIP file")
        print("   The dataset format may have changed")
        print("   Try manual download from UCI website")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = download_z_alizadeh_sani()
    
    if success:
        print("\n🎉 Dataset ready for processing!")
        print("Run: python data/process_z_alizadeh_sani.py")
    else:
        print("\n⚠️  Download failed. Try manual download from:")
        print("https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani")
