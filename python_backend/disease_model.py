import numpy as np
from PIL import Image
import io

def analyze_crop_image(image_bytes: bytes) -> dict:
    """Analyze crop foliage image using color-space thresholding to diagnose leaf diseases."""
    try:
        # Load image with Pillow
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img.resize((150, 150))  # Resize for quick calculations
        img_np = np.array(img_resized)
        
        # Reshape to pixels list
        pixels = img_np.reshape(-1, 3)
        total_pixels = len(pixels)
        
        green_count = 0
        yellow_count = 0
        brown_count = 0
        
        # Threshold pixels based on color channels
        # Green: G > R + 10 and G > B + 10 (Dominant green)
        # Yellow/Light Green: R and G are high, B is low (R > 130, G > 130, B < 120)
        # Brown/Necrotic: R and G are moderate, B is low (R > 80, G > 40, B < 60, R > G)
        for r, g, b in pixels:
            r, g, b = int(r), int(g), int(b)
            if g > r + 5 and g > b + 5:
                green_count += 1
            elif r > 120 and g > 110 and b < 100:
                yellow_count += 1
            elif r > 70 and g > 30 and b < 60 and r > g:
                brown_count += 1
                
        # Calculate percentages
        green_pct = (green_count / total_pixels) * 100
        yellow_pct = (yellow_count / total_pixels) * 100
        brown_pct = (brown_count / total_pixels) * 100
        other_pct = 100.0 - (green_pct + yellow_pct + brown_pct)
        
        # Calculate severity index (proportion of anomalous areas over foliage)
        anomaly_pixels = yellow_count + brown_count
        severity_pct = (anomaly_pixels / total_pixels) * 100 * 2.0  # Scale multiplier
        severity_pct = min(100.0, round(severity_pct, 1))
        
        # Diagnostic categorization
        if severity_pct < 10.0:
            status = "Healthy"
            disease = "No disease detected. Canopy looks healthy."
            treatment = "Continue normal watering and organic nutrition."
            severity_level = "Negligible"
        elif brown_pct > yellow_pct and brown_pct > 5.0:
            status = "Diseased"
            disease = "Leaf Spot / Fungal Leaf Blast"
            treatment = "Remove infected leaves. Spray copper oxychloride (2g/L) or a organic neem-based extract. Ensure proper plant spacing to decrease humidity."
            severity_level = "Severe" if severity_pct > 40.0 else "Moderate"
        elif yellow_pct > 5.0:
            status = "Diseased / Deficient"
            disease = "Chlorosis (Mosaic Virus or Nitrogen Deficiency)"
            treatment = "Apply urea or nitrogen-rich fertilizers if it is a deficiency. If mosaic virus is suspected, control vector insects like whiteflies using insecticidal soap."
            severity_level = "Severe" if severity_pct > 40.0 else "Moderate"
        else:
            status = "Under Stress"
            disease = "Early-stage stress or dehydration"
            treatment = "Monitor irrigation scheduling. Soil moisture may be insufficient."
            severity_level = "Mild"
            
        return {
            "status": status,
            "detected_disease": disease,
            "severity_percentage": severity_pct,
            "severity_level": severity_level,
            "recommended_action": treatment,
            "color_signature": {
                "healthy_green_pct": round(green_pct, 1),
                "chlorosis_yellow_pct": round(yellow_pct, 1),
                "necrotic_brown_pct": round(brown_pct, 1),
                "other_shadow_pct": round(other_pct, 1)
            }
        }
    except Exception as e:
        print(f"Local Image Model Error: {e}")
        return {
            "status": "Error",
            "detected_disease": "Could not execute local image modeling.",
            "severity_percentage": 0.0,
            "severity_level": "Unknown",
            "recommended_action": "Verify file upload format.",
            "color_signature": {}
        }
